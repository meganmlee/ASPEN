"""
TSformer-SA: A Temporal-Spectral Fusion Transformer with Subject-Specific Adapter for EEG Classification

Reference: Li et al. (2025) - A temporal–spectral fusion transformer with subject-specific adapter for enhancing RSVP-BCI decoding
Paper: https://doi.org/10.1016/j.neunet.2024.106844
GitHub: https://github.com/lixujin99/TSformer-SA

Architecture:
1. Dual-stream feature extraction (Temporal + Spectral views)
2. Cross-view interaction module (cross-attention + token fusion)
3. Attention-based fusion module
4. Subject-specific adapter for fine-tuning

Supports: SSVEP, P300, MI (Motor Imagery), Imagined Speech, Lee2019_MI, Lee2019_SSVEP, BNCI2014_P300
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
from typing import Optional, Dict, Tuple, List

# Add scale-net directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scale-net'))

try:
    from seed_utils import seed_everything, worker_init_fn, get_generator
    from dataset import load_dataset, TASK_CONFIGS
    DATASET_AVAILABLE = True
except ImportError:
    print("Warning: seed_utils/dataset not found. Using standalone mode.")
    DATASET_AVAILABLE = False

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    print("Warning: einops not installed. Install with: pip install einops")
    EINOPS_AVAILABLE = False


# ==================== Spectral Feature Extraction ====================

class SpectrogramExtractor(nn.Module):
    """
    Extract spectrogram features from raw EEG signals using Short-Time Fourier Transform (STFT)
    This converts temporal EEG to spectral view (time-frequency representation)
    """
    def __init__(self, n_fft: int = 64, hop_length: int = 16, 
                 freq_bins: int = 32, normalize: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = freq_bins
        self.normalize = normalize
        
        # Hanning window
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - raw EEG signals
        Returns:
            spectrogram: (B, C, F, T') - time-frequency representation
        """
        B, C, T = x.shape
        
        # Process each channel
        specs = []
        for c in range(C):
            # STFT for each channel
            spec = torch.stft(
                x[:, c, :], 
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode='reflect'
            )
            # Magnitude spectrogram
            spec_mag = torch.abs(spec)  # (B, F, T')
            specs.append(spec_mag)
        
        # Stack channels: (B, C, F, T')
        spectrogram = torch.stack(specs, dim=1)
        
        # Take only freq_bins (low frequency components are most relevant for EEG)
        spectrogram = spectrogram[:, :, :self.freq_bins, :]
        
        # Normalize
        if self.normalize:
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        
        return spectrogram


class LearnableSpectrogramExtractor(nn.Module):
    """
    Learnable spectrogram extraction using 1D convolutions with different filter sizes
    to capture multi-scale frequency information
    """
    def __init__(self, n_channels: int, n_samples: int, 
                 n_filters: int = 32, output_freq_bins: int = 32):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.output_freq_bins = output_freq_bins
        
        # Multi-scale temporal convolutions to capture different frequency bands
        # Different kernel sizes approximate different frequency bands
        filter_sizes = [4, 8, 16, 32, 64]
        self.freq_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, n_filters, kernel_size=k, stride=k//2, padding=k//4),
                nn.BatchNorm1d(n_filters),
                nn.ELU()
            ) for k in filter_sizes if k <= n_samples // 4
        ])
        
        # Project to desired frequency bins
        n_bands = len(self.freq_convs)
        self.projection = nn.Conv2d(n_bands * n_filters, output_freq_bins, kernel_size=1)
        self.norm = nn.LayerNorm(output_freq_bins)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - raw EEG signals
        Returns:
            (B, F, T') - learned spectral representation
        """
        B, C, T = x.shape
        
        # Apply multi-scale convolutions
        freq_features = []
        for conv in self.freq_convs:
            feat = conv(x)  # (B, n_filters, T')
            freq_features.append(feat)
        
        # Align temporal dimensions (take minimum)
        min_t = min(f.shape[-1] for f in freq_features)
        freq_features = [f[:, :, :min_t] for f in freq_features]
        
        # Stack along channel dim: (B, n_bands * n_filters, T')
        stacked = torch.cat(freq_features, dim=1)
        
        # Reshape for projection: (B, n_bands * n_filters, 1, T')
        stacked = stacked.unsqueeze(2)
        
        # Project to freq bins: (B, output_freq_bins, 1, T')
        out = self.projection(stacked)
        
        # Remove dummy dim: (B, F, T')
        out = out.squeeze(2)
        
        return out


# ==================== Temporal Patch Embedding ====================

class TemporalPatchEmbedding(nn.Module):
    """
    Patch embedding for temporal (raw EEG) view
    Uses CNN to extract local temporal features and create patches
    """
    def __init__(self, n_channels: int, n_samples: int, 
                 emb_size: int = 64, patch_size: int = 25,
                 dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.emb_size = emb_size
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),  # Temporal filter
            nn.BatchNorm2d(40),
            nn.ELU(),
        )
        
        # Spatial convolution (depthwise)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, (n_channels, 1), groups=1),  # Spatial filter
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, patch_size)),  # Create patches via pooling
            nn.Dropout(dropout),
        )
        
        # Calculate output sequence length
        self.n_patches = n_samples // patch_size
        
        # Project to embedding size
        self.projection = nn.Linear(40, emb_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - raw EEG
        Returns:
            (B, n_patches, emb_size) - patch embeddings
        """
        # Add channel dim: (B, 1, C, T)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.temporal_conv(x)  # (B, 40, C, T)
        x = self.spatial_conv(x)    # (B, 40, 1, n_patches)
        
        # Reshape: (B, n_patches, 40)
        x = x.squeeze(2).permute(0, 2, 1)
        
        # Project: (B, n_patches, emb_size)
        x = self.projection(x)
        
        return x


# ==================== Spectral Patch Embedding ====================

class SpectralPatchEmbedding(nn.Module):
    """
    Patch embedding for spectral (spectrogram) view
    Treats spectrogram as 2D image and extracts patch features
    """
    def __init__(self, n_channels: int, freq_bins: int, time_frames: int,
                 emb_size: int = 64, patch_size: Tuple[int, int] = (8, 8),
                 dropout: float = 0.1):
        super().__init__()
        
        self.emb_size = emb_size
        self.patch_size = patch_size
        
        # 2D convolution for spectral feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(patch_size),  # Create patches
            nn.Dropout(dropout),
        )
        
        # Calculate number of patches
        self.n_patches_freq = freq_bins // patch_size[0]
        self.n_patches_time = time_frames // patch_size[1]
        self.n_patches = self.n_patches_freq * self.n_patches_time
        
        # Project to embedding size
        self.projection = nn.Linear(64, emb_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F, T) - spectrogram
        Returns:
            (B, n_patches, emb_size) - patch embeddings
        """
        # CNN feature extraction
        x = self.conv_layers(x)  # (B, 64, F', T')
        
        # Reshape to patches: (B, 64, n_patches)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (B, 64, n_patches)
        
        # Permute: (B, n_patches, 64)
        x = x.permute(0, 2, 1)
        
        # Project: (B, n_patches, emb_size)
        x = self.projection(x)
        
        return x


# ==================== Positional Encoding ====================

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, emb_size: int, max_length: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, max_length, emb_size) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = x + self.encoding[:, :seq_len, :]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed)"""
    def __init__(self, emb_size: int, max_length: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_length, emb_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


# ==================== Multi-Head Attention ====================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        
        self.qkv = nn.Linear(emb_size, 3 * emb_size)
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-Attention between two modalities"""
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N1, C) - query tokens
            key_value: (B, N2, C) - key/value tokens from other modality
        Returns:
            (B, N1, C) - cross-attended features
        """
        B, N1, C = query.shape
        N2 = key_value.shape[1]
        
        # Project
        q = self.q_proj(query).reshape(B, N1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        out = self.out_proj(out)
        
        return out


# ==================== Feed-Forward Network ====================

class FeedForward(nn.Module):
    """Feed-Forward Network with GELU activation"""
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_size = emb_size * expansion
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== Transformer Encoder Block ====================

class TransformerEncoderBlock(nn.Module):
    """Standard Transformer Encoder Block"""
    def __init__(self, emb_size: int, num_heads: int = 8, 
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # FFN with residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ==================== Cross-View Interaction Module ====================

class CrossViewInteractionModule(nn.Module):
    """
    Cross-View Interaction Module from TSformer-SA
    Facilitates information transfer between temporal and spectral views
    using cross-attention and token fusion mechanisms
    """
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Cross-attention: temporal -> spectral
        self.cross_attn_t2s = CrossAttention(emb_size, num_heads, dropout)
        # Cross-attention: spectral -> temporal
        self.cross_attn_s2t = CrossAttention(emb_size, num_heads, dropout)
        
        # Layer norms
        self.norm_t = nn.LayerNorm(emb_size)
        self.norm_s = nn.LayerNorm(emb_size)
        
        # Token fusion MLPs
        self.fusion_mlp_t = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_mlp_s = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_feat: torch.Tensor, 
                spectral_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            temporal_feat: (B, N_t, C) - temporal view features
            spectral_feat: (B, N_s, C) - spectral view features
        Returns:
            enhanced_temporal: (B, N_t, C)
            enhanced_spectral: (B, N_s, C)
        """
        # Cross-attention
        t_from_s = self.cross_attn_t2s(self.norm_t(temporal_feat), self.norm_s(spectral_feat))
        s_from_t = self.cross_attn_s2t(self.norm_s(spectral_feat), self.norm_t(temporal_feat))
        
        # Token fusion: concatenate and fuse
        temporal_concat = torch.cat([temporal_feat, t_from_s], dim=-1)
        spectral_concat = torch.cat([spectral_feat, s_from_t], dim=-1)
        
        enhanced_temporal = temporal_feat + self.dropout(self.fusion_mlp_t(temporal_concat))
        enhanced_spectral = spectral_feat + self.dropout(self.fusion_mlp_s(spectral_concat))
        
        return enhanced_temporal, enhanced_spectral


# ==================== Attention-Based Fusion Module ====================

class AttentionFusionModule(nn.Module):
    """
    Attention-based Fusion Module from TSformer-SA
    Fuses temporal and spectral features using attention mechanism
    """
    def __init__(self, emb_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Learnable fusion query token
        self.fusion_query = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        
        # Multi-head attention for fusion
        self.fusion_attn = MultiHeadAttention(emb_size, num_heads, dropout)
        
        # FFN after fusion
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = FeedForward(emb_size, expansion=4, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, temporal_feat: torch.Tensor, 
                spectral_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_feat: (B, N_t, C)
            spectral_feat: (B, N_s, C)
        Returns:
            fused_feat: (B, C) - fused feature vector
        """
        B = temporal_feat.shape[0]
        
        # Concatenate features from both views
        combined = torch.cat([temporal_feat, spectral_feat], dim=1)  # (B, N_t + N_s, C)
        
        # Add fusion query token
        fusion_q = self.fusion_query.expand(B, -1, -1)  # (B, 1, C)
        combined_with_query = torch.cat([fusion_q, combined], dim=1)  # (B, 1 + N_t + N_s, C)
        
        # Self-attention over combined features
        attended = combined_with_query + self.dropout(
            self.fusion_attn(self.norm1(combined_with_query))
        )
        
        # FFN
        attended = attended + self.dropout(self.ffn(self.norm2(attended)))
        
        # Extract fusion token as final representation
        fused_feat = attended[:, 0, :]  # (B, C)
        
        return fused_feat


# ==================== Subject-Specific Adapter ====================

class SubjectSpecificAdapter(nn.Module):
    """
    Subject-Specific Adapter from TSformer-SA
    Lightweight adapter for rapid fine-tuning to new subjects
    Only this module is trained during fine-tuning stage
    """
    def __init__(self, emb_size: int, reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        
        hidden_size = emb_size // reduction
        
        # Bottleneck adapter
        self.adapter = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout),
        )
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) or (B, C) - input features
        Returns:
            adapted features with same shape
        """
        return x + self.scale * self.adapter(x)


# ==================== Multi-View Consistency Loss ====================

class MultiViewConsistencyLoss(nn.Module):
    """
    Multi-View Consistency Loss from TSformer-SA
    Maximizes feature similarity between temporal and spectral views
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, temporal_feat: torch.Tensor, 
                spectral_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive consistency loss between views
        
        Args:
            temporal_feat: (B, C) - global temporal features
            spectral_feat: (B, C) - global spectral features
        Returns:
            consistency_loss: scalar
        """
        # Normalize features
        temporal_feat = F.normalize(temporal_feat, dim=-1)
        spectral_feat = F.normalize(spectral_feat, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(temporal_feat, spectral_feat.T) / self.temperature
        
        # Diagonal elements are positive pairs
        B = temporal_feat.shape[0]
        labels = torch.arange(B, device=temporal_feat.device)
        
        # Cross-entropy loss (both directions)
        loss_t2s = F.cross_entropy(sim_matrix, labels)
        loss_s2t = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_t2s + loss_s2t) / 2


# ==================== TSformer-SA Main Model ====================

class TSformerSA(nn.Module):
    """
    TSformer-SA: Temporal-Spectral Fusion Transformer with Subject-Specific Adapter
    
    Architecture:
    1. Dual-stream feature extraction (temporal + spectral)
    2. Transformer encoders for each view
    3. Cross-view interaction module
    4. Attention-based fusion module  
    5. Subject-specific adapter (for fine-tuning)
    6. Classification head
    """
    def __init__(self, 
                 n_channels: int = 64,
                 n_samples: int = 250,
                 n_classes: int = 2,
                 # Embedding parameters
                 emb_size: int = 64,
                 # Transformer parameters
                 num_heads: int = 8,
                 num_layers: int = 4,
                 # Spectral parameters
                 n_fft: int = 64,
                 hop_length: int = 16,
                 freq_bins: int = 32,
                 # Dropout
                 dropout: float = 0.1,
                 # Adapter
                 use_adapter: bool = True,
                 adapter_reduction: int = 4):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.is_binary = (n_classes == 2)
        self.use_adapter = use_adapter
        
        # ===== Spectral View Processing =====
        # Spectrogram extraction
        self.spectrogram_extractor = SpectrogramExtractor(
            n_fft=n_fft,
            hop_length=hop_length,
            freq_bins=freq_bins,
            normalize=True
        )
        
        # Calculate spectral dimensions
        time_frames = (n_samples + n_fft) // hop_length + 1
        
        # Spectral patch embedding
        self.spectral_patch_embed = nn.Sequential(
            nn.Conv2d(n_channels, emb_size, kernel_size=(freq_bins, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            Rearrange('b c 1 t -> b t c') if EINOPS_AVAILABLE else nn.Flatten(2, 3),
        )
        
        # ===== Temporal View Processing =====
        # Temporal patch embedding (EEGNet-style)
        patch_size = max(n_samples // 16, 10)
        self.temporal_patch_embed = TemporalPatchEmbedding(
            n_channels=n_channels,
            n_samples=n_samples,
            emb_size=emb_size,
            patch_size=patch_size,
            dropout=dropout
        )
        
        # ===== Positional Encodings =====
        self.temporal_pos_enc = LearnablePositionalEncoding(emb_size, max_length=200, dropout=dropout)
        self.spectral_pos_enc = LearnablePositionalEncoding(emb_size, max_length=200, dropout=dropout)
        
        # ===== Transformer Encoders =====
        # Temporal stream encoder
        self.temporal_encoder = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, expansion=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Spectral stream encoder
        self.spectral_encoder = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, expansion=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # ===== Cross-View Interaction =====
        # Apply cross-view interaction at middle layers
        self.cross_view_layers = [num_layers // 2]  # Interact at middle
        self.cross_view_modules = nn.ModuleDict({
            str(i): CrossViewInteractionModule(emb_size, num_heads, dropout)
            for i in self.cross_view_layers
        })
        
        # ===== Fusion Module =====
        self.fusion_module = AttentionFusionModule(emb_size, num_heads, dropout)
        
        # ===== Subject-Specific Adapter =====
        if use_adapter:
            self.adapter_temporal = SubjectSpecificAdapter(emb_size, adapter_reduction, dropout)
            self.adapter_spectral = SubjectSpecificAdapter(emb_size, adapter_reduction, dropout)
            self.adapter_fusion = SubjectSpecificAdapter(emb_size, adapter_reduction, dropout)
        
        # ===== Classification Head =====
        self.norm = nn.LayerNorm(emb_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_size, 1 if self.is_binary else n_classes)
        )
        
        # ===== Multi-View Consistency Loss =====
        self.consistency_loss_fn = MultiViewConsistencyLoss(temperature=0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def freeze_pretrained(self):
        """Freeze all parameters except adapters (for fine-tuning stage)"""
        for param in self.parameters():
            param.requires_grad = False
        
        if self.use_adapter:
            for param in self.adapter_temporal.parameters():
                param.requires_grad = True
            for param in self.adapter_spectral.parameters():
                param.requires_grad = True
            for param in self.adapter_fusion.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters (for pre-training stage)"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_consistency_loss(self, temporal_feat: torch.Tensor, 
                             spectral_feat: torch.Tensor) -> torch.Tensor:
        """Compute multi-view consistency loss"""
        # Global pooling
        t_global = temporal_feat.mean(dim=1)  # (B, C)
        s_global = spectral_feat.mean(dim=1)  # (B, C)
        return self.consistency_loss_fn(t_global, s_global)
    
    def forward(self, x: torch.Tensor, 
                return_consistency_loss: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, C, T) or (B, 1, C, T) - raw EEG signals
            return_consistency_loss: if True, also return consistency loss
            
        Returns:
            logits: (B, n_classes) or (B, 1) for binary
            consistency_loss: (optional) multi-view consistency loss
        """
        # Handle input dimensions
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove channel dim if present
        
        B, C, T = x.shape
        
        # ===== Spectral View =====
        # Extract spectrogram
        spectrogram = self.spectrogram_extractor(x)  # (B, C, F, T')
        
        # Spectral patch embedding
        spectral_feat = self.spectral_patch_embed(spectrogram)  # (B, T', emb_size)
        if not EINOPS_AVAILABLE:
            # Manual reshape if einops not available
            spectral_feat = spectral_feat.permute(0, 2, 1)
        
        spectral_feat = self.spectral_pos_enc(spectral_feat)
        
        # ===== Temporal View =====
        temporal_feat = self.temporal_patch_embed(x)  # (B, n_patches, emb_size)
        temporal_feat = self.temporal_pos_enc(temporal_feat)
        
        # ===== Dual-Stream Transformer Encoding =====
        for i in range(len(self.temporal_encoder)):
            # Temporal encoder block
            temporal_feat = self.temporal_encoder[i](temporal_feat)
            
            # Spectral encoder block
            spectral_feat = self.spectral_encoder[i](spectral_feat)
            
            # Cross-view interaction at specified layers
            if i in self.cross_view_layers:
                temporal_feat, spectral_feat = self.cross_view_modules[str(i)](
                    temporal_feat, spectral_feat
                )
        
        # ===== Apply Adapters =====
        if self.use_adapter:
            temporal_feat = self.adapter_temporal(temporal_feat)
            spectral_feat = self.adapter_spectral(spectral_feat)
        
        # ===== Compute Consistency Loss (if needed) =====
        consistency_loss = None
        if return_consistency_loss:
            consistency_loss = self.get_consistency_loss(temporal_feat, spectral_feat)
        
        # ===== Fusion =====
        fused_feat = self.fusion_module(temporal_feat, spectral_feat)  # (B, emb_size)
        
        # Apply fusion adapter
        if self.use_adapter:
            fused_feat = self.adapter_fusion(fused_feat)
        
        # ===== Classification =====
        fused_feat = self.norm(fused_feat)
        logits = self.classifier(fused_feat)
        
        if return_consistency_loss:
            return logits, consistency_loss
        return logits


# ==================== Dataset ====================

class PreprocessedEEGDataset(Dataset):
    """Dataset for EEG data that has already been preprocessed."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        # Convert to float32 immediately to save memory/time during training
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simply return the pre-existing signal and label
        return self.data[idx], self.labels[idx]


# ==================== Data Loader Creation ====================

def create_dataloaders(datasets: Dict, batch_size: int = 32, 
                       num_workers: int = 4, seed: int = 44,
                       augment_train: bool = False) -> Dict: # Added augment_train here
    """Create DataLoaders using the simplified PreprocessedEEGDataset."""
    loaders = {}
    
    for split, (X, y) in datasets.items():
        shuffle = (split == 'train')
        
        # Note: If you want to use augment_train, you'd apply it to the 
        # training dataset here. For now, this just enables the code to run.
        ds = PreprocessedEEGDataset(X, y)
        
        loaders[split] = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True,
            worker_init_fn=(lambda worker_id: worker_init_fn(worker_id, seed)) if DATASET_AVAILABLE else None,
            generator=(get_generator(seed) if shuffle and DATASET_AVAILABLE else None)
        )
    
    return loaders


# ==================== Device Setup ====================

def setup_device():
    """Setup device and return device info"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        print(f"CUDA available: {n_gpus} GPU(s) detected")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return device, n_gpus
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu'), 0


def wrap_model_multi_gpu(model, n_gpus):
    """Wrap model with DataParallel if multiple GPUs available"""
    if n_gpus > 1:
        print(f"Using DataParallel with {n_gpus} GPUs")
        model = nn.DataParallel(model)
    return model


def unwrap_model(model):
    """Get the underlying model from DataParallel wrapper"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ==================== Training Functions ====================

def train_epoch(model, loader, criterion, optimizer, device, 
                is_binary: bool = False, use_consistency_loss: bool = True,
                consistency_weight: float = 0.1):
    """Train for one epoch"""
    model.train()
    total_loss, total_cls_loss, total_cons_loss = 0, 0, 0
    correct, total = 0, 0
    
    pbar = tqdm(loader, desc='Train', ncols=120)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if is_binary:
            labels_loss = labels.float().unsqueeze(1)
        else:
            labels_loss = labels
        
        optimizer.zero_grad()
        
        if use_consistency_loss:
            outputs, cons_loss = model(inputs, return_consistency_loss=True)
            
            # DataParallel returns a vector of losses; we must average them
            if cons_loss.dim() > 0:
                cons_loss = cons_loss.mean()
                
            cls_loss = criterion(outputs, labels_loss)
            loss = cls_loss + consistency_weight * cons_loss
            total_cons_loss += cons_loss.item()
        else:
            outputs = model(inputs)
            cls_loss = criterion(outputs, labels_loss)
            loss = cls_loss
        
        total_cls_loss += cls_loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        if is_binary:
            pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
        else:
            _, pred = outputs.max(1)
        
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}', 
            'cls': f'{total_cls_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device, criterion=None, is_binary: bool = False):
    """Evaluate model"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_binary:
                labels_loss = labels.float().unsqueeze(1)
            else:
                labels_loss = labels
            
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, labels_loss)
                total_loss += loss.item()
            
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader) if criterion is not None else None
    acc = 100. * correct / total
    return avg_loss, acc


# ==================== Configuration ====================

def get_tsformer_config(task: str, n_channels: int, n_samples: int, 
                        sampling_rate: int) -> Dict:
    """
    Get TSformer-SA hyperparameters optimized for each task
    """
    # Default configuration
    config = {
        'emb_size': 64,
        'num_heads': 8,
        'num_layers': 4,
        'n_fft': min(64, n_samples // 4),
        'hop_length': max(8, n_samples // 32),
        'freq_bins': 32,
        'dropout': 0.1,
        'use_adapter': True,
        'adapter_reduction': 4,
    }
    
    # Task-specific adjustments
    if task in ['SSVEP', 'Lee2019_SSVEP']:
        config['num_layers'] = 4
        config['freq_bins'] = 40  # More frequency resolution for SSVEP
        config['n_fft'] = min(128, n_samples // 2)
    elif task in ['P300', 'BNCI2014_P300']:
        config['num_layers'] = 3
        config['emb_size'] = 48
        config['num_heads'] = 4
    elif task in ['MI', 'Lee2019_MI']:
        config['num_layers'] = 4
        config['freq_bins'] = 32
    elif task == 'Imagined_speech':
        config['num_layers'] = 5
        config['emb_size'] = 96
        config['freq_bins'] = 48
    
    # Ensure num_heads divides emb_size
    while config['emb_size'] % config['num_heads'] != 0:
        config['num_heads'] -= 1
    
    # Ensure n_fft is valid
    config['n_fft'] = min(config['n_fft'], n_samples)
    
    return config


# ==================== Main Training ====================

def train_task(task: str, config: Optional[Dict] = None, 
               model_path: Optional[str] = None,
               fine_tune: bool = False,
               pretrained_path: Optional[str] = None) -> Tuple:
    """
    Train TSformer-SA for a specific EEG task
    
    Args:
        task: One of 'SSVEP', 'P300', 'MI', 'Imagined_speech', etc.
        config: Training configuration
        model_path: Path to save best model
        fine_tune: If True, only train adapters (requires pretrained_path)
        pretrained_path: Path to pretrained model for fine-tuning
        
    Returns:
        (model, results_dict)
    """
    if not DATASET_AVAILABLE:
        raise ImportError("Dataset utilities not available. Please ensure seed_utils and dataset modules are accessible.")
    
    task_config = TASK_CONFIGS.get(task, {})
    
    if config is None:
        config = {
            'data_dir': task_config.get('data_dir'),
            'num_seen': task_config.get('num_seen'),
            'seed': 44,
            'n_classes': task_config.get('num_classes', 2),
            'sampling_rate': task_config.get('sampling_rate', 250),
            'batch_size': 64,
            'num_epochs': 100,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,
            'consistency_weight': 0.1,
        }
    else:
        config.setdefault('n_classes', task_config.get('num_classes', 2))
        config.setdefault('sampling_rate', task_config.get('sampling_rate', 250))
        config.setdefault('consistency_weight', 0.1)
        config.setdefault('data_dir', task_config.get('data_dir'))
        config.setdefault('num_seen', task_config.get('num_seen'))
    
    seed = config.get('seed', 44)
    seed_everything(seed, deterministic=True)
    
    device, n_gpus = setup_device()
    
    print(f"\n{'='*70}")
    print(f"TSformer-SA - {task} Classification")
    print(f"{'='*70}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    print(f"Mode: {'Fine-tuning' if fine_tune else 'Pre-training'}")
    
    # ====== Load Data ======
    datasets = load_dataset(
        task=task,
        data_dir=config.get('data_dir'),
        num_seen=config.get('num_seen'),
        seed=config.get('seed', 44)
    )
    
    if not datasets:
        raise ValueError(f"Failed to load data for task: {task}")
    
    # ====== Create Data Loaders ======
    loaders = create_dataloaders(
        datasets, 
        batch_size=config['batch_size'],
        num_workers=4,
        augment_train=True,
        seed=seed
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    # Get dimensions
    sample_x, _ = next(iter(train_loader))
    n_channels, n_samples = sample_x.shape[1], sample_x.shape[2]
    print(f"Input shape: ({n_channels} channels, {n_samples} samples)")
    
    # ====== Get TSformer-SA Configuration ======
    tsformer_config = get_tsformer_config(
        task, n_channels, n_samples, config['sampling_rate']
    )
    print(f"\nTSformer-SA Configuration:")
    for k, v in tsformer_config.items():
        print(f"  {k}: {v}")
    
    # ====== Create Model ======
    n_classes = config['n_classes']
    model = TSformerSA(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        **tsformer_config
    ).to(device)
    
    # Load pretrained weights if fine-tuning
    if fine_tune and pretrained_path:
        print(f"\nLoading pretrained model from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        unwrap_model(model).freeze_pretrained()
        print("Frozen pretrained weights, only adapters will be trained")
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {n_params:,}")
    print(f"Trainable Parameters: {n_trainable:,}")
    print(f"Classes: {n_classes}")
    
    model = wrap_model_multi_gpu(model, n_gpus)
    
    # ====== Loss & Optimizer ======
    is_binary = (n_classes == 2)
    train_labels = datasets['train'][1]
    
    if is_binary:
        class_counts = np.bincount(train_labels)
        class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        print(f"  Imbalance Ratio: {class_ratio:.2f}:1")
        
        if class_ratio > 1.5 or class_ratio < 0.67:
            pos_weight = torch.tensor([class_ratio], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using BCEWithLogitsLoss with pos_weight={class_ratio:.2f}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using BCEWithLogitsLoss without pos_weight")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss for {n_classes}-class classification")
    
    # Different LR for fine-tuning
    lr = config['lr'] * 0.1 if fine_tune else config['lr']
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    
    # ====== Training Loop ======
    best_val_acc = 0
    patience_counter = 0
    
    if model_path is None:
        mode = 'finetune' if fine_tune else 'pretrain'
        model_path = f'best_tsformer_sa_{task.lower()}_{mode}_model.pth'
    
    # Use consistency loss only during pre-training
    use_consistency_loss = not fine_tune
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            is_binary=is_binary,
            use_consistency_loss=use_consistency_loss,
            consistency_weight=config['consistency_weight']
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion, is_binary=is_binary)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'task': task,
                'config': config,
                'tsformer_config': tsformer_config,
                'n_channels': n_channels,
                'n_samples': n_samples,
                'fine_tuned': fine_tune,
            }, model_path)
            print(f"✓ Best model saved! ({val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print("\nEarly stopping triggered!")
            break
    
    # ====== Final Evaluation ======
    print(f"\n{'='*70}")
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(model_path)
    unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    
    results = {'val': best_val_acc}
    
    if test1_loader:
        test1_loss, test1_acc = evaluate(model, test1_loader, device, criterion, is_binary=is_binary)
        results['test1'] = test1_acc
        results['test1_loss'] = test1_loss
    
    if test2_loader:
        test2_loss, test2_acc = evaluate(model, test2_loader, device, criterion, is_binary=is_binary)
        results['test2'] = test2_acc
        results['test2_loss'] = test2_loss
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {task} (TSformer-SA)")
    print(f"{'='*70}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    if 'test1' in results:
        print(f"Test1 (Seen):    {results['test1']:.2f}% (loss {results['test1_loss']:.4f})")
    if 'test2' in results:
        print(f"Test2 (Unseen):  {results['test2']:.2f}% (loss {results['test2_loss']:.4f})")
    print(f"{'='*70}")
    
    return model, results


def train_all_tasks(tasks: Optional[list] = None, save_dir: str = './checkpoints'):
    """Train TSformer-SA models for all specified tasks"""
    if tasks is None:
        tasks = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300']
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    print("=" * 80)
    print("TSformer-SA - MULTI-TASK EEG CLASSIFICATION")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")
        
        try:
            model_path = os.path.join(save_dir, f'best_tsformer_sa_{task.lower()}_model.pth')
            model, results = train_task(task, model_path=model_path)
            all_results[task] = results
            
            print(f"\n{task} completed!")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc: {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc: {results['test2']:.2f}%")
                
        except Exception as e:
            print(f"Error training {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS (TSformer-SA)")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        if 'error' in results:
            print(f"\n{task}: FAILED - {results['error']}")
        else:
            print(f"\n{task}:")
            print(f"  Best Val Acc: {results['val']:.2f}%")
            if 'test1' in results:
                print(f"  Test1 Acc:    {results['test1']:.2f}%")
            if 'test2' in results:
                print(f"  Test2 Acc:    {results['test2']:.2f}%")
    
    print(f"\n{'='*80}")
    print("TSformer-SA MULTI-TASK TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    return all_results


# ==================== Standalone Test ====================

def test_model_standalone():
    """Test model with random data (no dataset dependencies)"""
    print("="*60)
    print("TSformer-SA Standalone Test")
    print("="*60)
    
    # Test parameters
    batch_size = 4
    n_channels = 64
    n_samples = 250
    n_classes = 2
    
    # Create model
    model = TSformerSA(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        emb_size=64,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        use_adapter=True,
    )
    
    print(f"\nModel created successfully!")
    print(f"Input shape: (B, {n_channels}, {n_samples})")
    print(f"Output shape: (B, 1) for binary classification")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, n_channels, n_samples)
    
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test with consistency loss
    print("\nTesting forward pass with consistency loss...")
    model.train()
    output, cons_loss = model(x, return_consistency_loss=True)
    print(f"Output shape: {output.shape}")
    print(f"Consistency loss: {cons_loss.item():.4f}")
    
    # Test adapter freezing
    print("\nTesting adapter freezing for fine-tuning...")
    model.freeze_pretrained()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {n_trainable:,}")
    
    model.unfreeze_all()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {n_trainable:,}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TSformer-SA on EEG tasks')
    parser.add_argument('--task', type=str, default='SSVEP',
                        choices=['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 
                                 'Lee2019_SSVEP', 'BNCI2014_P300', 'all', 'BI2014b_P300'],
                        help='Task to train on (default: SSVEP, use "test" for standalone test)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=44,
                        help='Random seed')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune mode (only train adapters)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model for fine-tuning')
    
    args = parser.parse_args()
    
    if args.task == 'test':
        test_model_standalone()
    else:
        config = {
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': 1e-4,
            'patience': 20,
            'consistency_weight': 0.1,
            'seed': args.seed,
        }
        
        if args.task == 'all':
            results = train_all_tasks(save_dir=args.save_dir)
        else:
            model_path = os.path.join(args.save_dir, f'best_tsformer_sa_{args.task.lower()}_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            model, results = train_task(
                args.task, 
                config=config, 
                model_path=model_path,
                fine_tune=args.fine_tune,
                pretrained_path=args.pretrained
            )