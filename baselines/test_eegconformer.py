"""
Test script for EEGConformer Baseline Model
Evaluates trained EEGConformer models on validation and test sets
"""

import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
from sklearn.metrics import confusion_matrix

# Add model directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from train_eegconformer import EEGConformer, create_raw_dataloaders
from dataset import load_dataset, TASK_CONFIGS


# ==================== Evaluation ====================

def evaluate(model, loader, device, is_binary=False):
    """
    Evaluate model on a dataset
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            if is_binary:
                pred = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return 100. * correct / total, np.array(all_preds), np.array(all_labels)


# ==================== Confusion Matrix ====================

def print_confusion_matrix(y_true, y_pred, n_classes, title="Confusion Matrix"):
    """Print confusion matrix as text in terminal"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    if n_classes == 2:
        class_labels = ['Non-Target', 'Target']
    elif n_classes == 4:
        class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    elif n_classes == 26:
        class_labels = [chr(65 + i) for i in range(n_classes)]
    else:
        class_labels = [str(i) for i in range(n_classes)]
    
    print(f"\n{title}")
    print("=" * 60)
    
    if n_classes > 10:
        print(f"(Confusion matrix too large to display, showing summary only)")
    else:
        header = "True\\Pred"
        for label in class_labels:
            header += f"{label:>8s}"
        header += "   Total"
        print(header)
        print("-" * len(header))
        
        for i in range(n_classes):
            row = f"{class_labels[i]:>9s}"
            total_true = cm[i].sum()
            for j in range(n_classes):
                row += f"{cm[i, j]:>8d}"
            row += f"{total_true:>8d}"
            print(row)
        
        print("-" * len(header))
        
        col_totals = cm.sum(axis=0)
        row = "    Total"
        for total in col_totals:
            row += f"{total:>8d}"
        row += f"{len(y_true):>8d}"
        print(row)
    
    print("=" * 60)
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = 100. * correct / total
    
    print(f"\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall accuracy: {accuracy:.2f}%")
    
    print(f"\nPer-class accuracy:")
    class_acc = []
    for i in range(n_classes):
        cls_total = cm[i].sum()
        cls_correct = cm[i, i]
        cls_acc_val = 100. * cls_correct / cls_total if cls_total > 0 else 0.0
        class_acc.append((i, cls_acc_val, cls_correct, cls_total))
    
    if n_classes > 10:
        class_acc_sorted = sorted(class_acc, key=lambda x: x[1])
        print("  Worst 5 classes:")
        for i, acc, correct, total in class_acc_sorted[:5]:
            print(f"    Class {i:2d} ({class_labels[i]:>3s}): {acc:5.1f}% ({correct}/{total})")
        print("  Best 5 classes:")
        for i, acc, correct, total in class_acc_sorted[-5:]:
            print(f"    Class {i:2d} ({class_labels[i]:>3s}): {acc:5.1f}% ({correct}/{total})")
    else:
        for i, acc, correct, total in class_acc:
            print(f"  Class {i:2d} ({class_labels[i]}): {acc:5.1f}% ({correct}/{total})")


# ==================== Main Test ====================

def test_task(task: str, checkpoint_path: str, batch_size: int = 32):
    """Test EEGConformer model for a specific task"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Testing EEGConformer - Task: {task}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return {'task': task, 'error': 'Checkpoint not found'}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return {'task': task, 'error': f'Checkpoint loading failed: {e}'}
    
    saved_config = checkpoint.get('config', {})
    conformer_config = checkpoint.get('conformer_config', {})
    n_channels = checkpoint.get('n_channels')
    n_samples = checkpoint.get('n_samples')
    task_defaults = TASK_CONFIGS.get(task, {})
    
    config = {
        'num_seen': saved_config.get('num_seen', task_defaults.get('num_seen', 33)),
        'seed': saved_config.get('seed', 44),
        'data_dir': saved_config.get('data_dir', task_defaults.get('data_dir', './data')),
        'n_classes': saved_config.get('n_classes', task_defaults.get('num_classes', 26)),
    }
    
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data Dir: {config['data_dir']}")
    
    datasets = load_dataset(
        task=task,
        data_dir=config['data_dir'],
        num_seen=config['num_seen'],
        seed=config['seed']
    )
    
    loaders = create_raw_dataloaders(
        datasets,
        batch_size=batch_size,
        num_workers=4,
        augment_train=False,
        seed=config['seed']
    )
    
    val_loader = loaders['val']
    test1_loader = loaders.get('test1')
    test2_loader = loaders.get('test2')
    
    if n_channels is None or n_samples is None:
        sample_x, _ = next(iter(val_loader))
        n_channels = sample_x.shape[1]
        n_samples = sample_x.shape[2]
    
    n_classes = config['n_classes']
    is_binary = (n_classes == 2)
    
    if not conformer_config:
        conformer_config = {
            'embed_dim': 40,
            'n_heads': 10,
            'n_layers': 6,
            'dim_ff': 256,
            'kernel_size': 25,
            'pool_size': 75,
            'pool_stride': 15,
            'dropout': 0.5,
            'emb_dropout': 0.5,
        }
    
    model = EEGConformer(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        **conformer_config
    ).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Checkpoint weights loaded successfully!")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    print(f"\n{'='*70}")
    print("Evaluating...")
    print(f"{'='*70}")
    
    results = {'task': task}
    
    val_acc, val_preds, val_labels = evaluate(model, val_loader, device, is_binary=is_binary)
    results['val_acc'] = val_acc
    
    if test1_loader:
        test1_acc, test1_preds, test1_labels = evaluate(model, test1_loader, device, is_binary=is_binary)
        results['test1_acc'] = test1_acc
    else:
        test1_acc, test1_preds, test1_labels = None, None, None
    
    if test2_loader:
        test2_acc, test2_preds, test2_labels = evaluate(model, test2_loader, device, is_binary=is_binary)
        results['test2_acc'] = test2_acc
    else:
        test2_acc, test2_preds, test2_labels = None, None, None
    
    print(f"\n{'='*70}")
    print(f"RESULTS - {task} (EEGConformer)")
    print(f"{'='*70}")
    print(f"Validation Acc:  {val_acc:.2f}%")
    if test1_acc is not None:
        print(f"Test1 (Seen):    {test1_acc:.2f}%")
    if test2_acc is not None:
        print(f"Test2 (Unseen):  {test2_acc:.2f}%")
    print(f"{'='*70}")
    
    if test1_preds is not None and test1_labels is not None:
        print_confusion_matrix(
            test1_labels, test1_preds, n_classes,
            title=f"Confusion Matrix - {task} Test1 (Seen Subjects)"
        )
    
    if test2_preds is not None and test2_labels is not None:
        print_confusion_matrix(
            test2_labels, test2_preds, n_classes,
            title=f"Confusion Matrix - {task} Test2 (Unseen Subjects)"
        )
    
    return results


def test_all_tasks(tasks: list, checkpoint_dir: str, batch_size: int = 32):
    """Test EEGConformer on all specified tasks"""
    all_results = {}
    
    print("\n" + "=" * 80)
    print("EEGConformer - Multi-Task Evaluation")
    print("=" * 80)
    
    for task in tasks:
        checkpoint_path = os.path.join(checkpoint_dir, f'best_conformer_{task.lower()}_model.pth')
        results = test_task(task, checkpoint_path, batch_size)
        all_results[task] = results
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - EEGConformer")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        print("-" * 40)
        if 'error' in results:
            print(f"🚨 {task}: FAILED - {results['error']}")
        else:
            print(f"✅ {task}:")
            print(f"   Val Acc:   {results['val_acc']:.2f}%")
            if 'test1_acc' in results:
                print(f"   Test1 Acc: {results['test1_acc']:.2f}% (Seen)")
            if 'test2_acc' in results:
                print(f"   Test2 Acc: {results['test2_acc']:.2f}% (Unseen)")
    
    print(f"\n{'='*80}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test EEGConformer model on EEG tasks"
    )
    
    tasks_available = ['SSVEP', 'P300', 'MI', 'Imagined_speech', 'Lee2019_MI', 'Lee2019_SSVEP', 'BNCI2014_P300']
    
    parser.add_argument(
        '--task', 
        type=str, 
        default='SSVEP',
        choices=tasks_available + ['all'],
        help=f"Task to test on. Choices: {tasks_available + ['all']} (default: SSVEP)"
    )
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        default='./checkpoints',
        help='Directory where model checkpoints are saved'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for testing (default: 32)'
    )
    
    args = parser.parse_args()
    
    if args.task == 'all':
        test_all_tasks(
            tasks=tasks_available, 
            checkpoint_dir=args.checkpoint_dir, 
            batch_size=args.batch_size
        )
    else:
        checkpoint_path = os.path.join(
            args.checkpoint_dir, 
            f'best_conformer_{args.task.lower()}_model.pth'
        )
        test_task(
            task=args.task, 
            checkpoint_path=checkpoint_path, 
            batch_size=args.batch_size
        )
