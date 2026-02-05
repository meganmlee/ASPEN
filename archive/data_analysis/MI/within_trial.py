from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import helpers

# Channel Selection Analysis
# Which channels should we keep?

# Output results
output_dir = Path('/home/megan/Downloads/cleaneddata/within_trial')
output_dir.mkdir(exist_ok=True)


def compute_channel_similarity(trial_data):
    """Compute correlation between all channel pairs."""
    return np.corrcoef(trial_data)


def identify_representative_channels(epochs, condition):
    """
    Identify which channels are most representative and should be kept.

    Returns rankings and recommendations for channel selection.
    """
    data = epochs[condition].get_data()  # (n_trials, n_channels, n_times)
    n_trials, n_channels, n_times = data.shape
    channel_names = epochs.ch_names

    # Compute average similarity across all trials
    trial_similarities = []
    for trial_idx in range(n_trials):
        sim_matrix = compute_channel_similarity(data[trial_idx])
        trial_similarities.append(sim_matrix)

    avg_similarity = np.mean(trial_similarities, axis=0)

    # Method 1: DIVERSITY SCORE - channels least correlated with others
    diversity_scores = []
    for i in range(n_channels):
        # Lower average correlation = more unique information
        avg_corr = np.mean(np.abs(avg_similarity[i, :]))
        diversity_score = 1 - avg_corr
        diversity_scores.append({
            'channel': channel_names[i],
            'diversity_score': diversity_score,
            'avg_correlation': avg_corr,
            'rank': 0  # Will be filled later
        })

    # Sort by diversity (highest = most distinct = keep first)
    diversity_scores = sorted(diversity_scores, key=lambda x: x['diversity_score'], reverse=True)
    for rank, item in enumerate(diversity_scores, 1):
        item['rank'] = rank

    # Method 2: CLUSTER REPRESENTATIVES - one channel per cluster
    distance_matrix = 1 - np.abs(avg_similarity)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    condensed_dist = squareform(distance_matrix)
    linkage_matrix = hierarchy.linkage(condensed_dist, method='average')

    # Get clusters at different thresholds
    cluster_results = {}
    for threshold in [0.3, 0.5, 0.7]:
        clusters = hierarchy.fcluster(linkage_matrix, t=threshold, criterion='distance')

        # Find best representative in each cluster (highest diversity)
        representatives = []
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_channels = [ch for i, ch in enumerate(channel_names) if cluster_mask[i]]

            # Get diversity scores for this cluster
            cluster_div = {d['channel']: d['diversity_score']
                          for d in diversity_scores if d['channel'] in cluster_channels}

            # Pick channel with highest diversity in this cluster
            best_channel = max(cluster_div, key=cluster_div.get)
            representatives.append(best_channel)

        cluster_results[f'threshold_{threshold}'] = {
            'n_clusters': len(np.unique(clusters)),
            'representatives': representatives
        }

    # Method 3: INCREMENTAL SELECTION - maximize diversity while adding channels
    selected_channels = []
    remaining = list(range(n_channels))

    # Start with most diverse channel
    first_idx = 0  # Already sorted by diversity
    selected_channels.append(diversity_scores[first_idx]['channel'])

    # Incrementally add channels that are least correlated with already selected
    for _ in range(min(14, n_channels - 1)):  # Select up to 15 total
        best_score = -1
        best_channel = None

        for d in diversity_scores:
            if d['channel'] in selected_channels:
                continue

            # Calculate average correlation with already selected channels
            ch_idx = channel_names.index(d['channel'])
            selected_indices = [channel_names.index(ch) for ch in selected_channels]
            avg_corr_with_selected = np.mean([np.abs(avg_similarity[ch_idx, si])
                                             for si in selected_indices])

            score = 1 - avg_corr_with_selected
            if score > best_score:
                best_score = score
                best_channel = d['channel']

        if best_channel:
            selected_channels.append(best_channel)

    return {
        'diversity_ranking': diversity_scores,
        'cluster_representatives': cluster_results,
        'incremental_selection': selected_channels,
        'avg_similarity_matrix': avg_similarity,
        'channel_names': channel_names,
        'linkage_matrix': linkage_matrix
    }


def plot_channel_selection_recommendations(results, subject_id, condition, save_path):
    """
    Visualize channel selection recommendations.
    """
    diversity_ranking = results['diversity_ranking']
    incremental = results['incremental_selection']
    cluster_reps = results['cluster_representatives']

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # Plot 1: Channel Ranking (Top 15)
    ax = fig.add_subplot(gs[0, :])
    top_15 = diversity_ranking[:15]
    channels = [d['channel'].replace('EEG-', '') for d in top_15]
    scores = [d['diversity_score'] for d in top_15]
    colors = ['darkgreen' if i < 5 else 'green' if i < 10 else 'yellowgreen'
              for i in range(len(channels))]

    bars = ax.bar(range(len(channels)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xlabel('Channel', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distinctiveness Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Top 15 Most Distinct Channels - {subject_id} ({condition})',
                fontsize=15, fontweight='bold')
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n#{i+1}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Add recommendation zones
    ax.axvspan(-0.5, 4.5, alpha=0.1, color='green', label='Top 5 (Must Keep)')
    ax.axvspan(4.5, 9.5, alpha=0.1, color='yellow', label='Top 10 (Recommended)')
    ax.axvspan(9.5, 14.5, alpha=0.1, color='orange', label='Top 15 (Optional)')
    ax.legend(fontsize=11, loc='upper right')

    # Plot 2: Incremental Selection Strategy
    ax = fig.add_subplot(gs[1, :2])
    incremental_15 = incremental[:15]
    inc_channels = [ch.replace('EEG-', '') for ch in incremental_15]

    y_pos = np.arange(len(inc_channels))
    ax.barh(y_pos, range(len(inc_channels), 0, -1), color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(inc_channels, fontsize=10)
    ax.set_xlabel('Selection Priority (higher = more important)', fontsize=12, fontweight='bold')
    ax.set_title('Incremental Selection Strategy\n(Maximizes diversity at each step)',
                fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add selection cutoff lines
    ax.axhline(4.5, color='red', linestyle='--', linewidth=2, label='Top 5')
    ax.axhline(9.5, color='orange', linestyle='--', linewidth=2, label='Top 10')
    ax.legend(fontsize=10)

    # Plot 3: Cluster-Based Selection Options
    ax = fig.add_subplot(gs[1, 2])

    thresholds = ['0.3\n(Aggressive)', '0.5\n(Balanced)', '0.7\n(Conservative)']
    n_channels = [cluster_reps[f'threshold_{t.split()[0]}']['n_clusters']
                  for t in ['0.3', '0.5', '0.7']]

    bars = ax.bar(range(len(thresholds)), n_channels,
                  color=['darkred', 'orange', 'lightblue'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Channels Selected', fontsize=12, fontweight='bold')
    ax.set_title('Cluster-Based Selection\n(One representative per group)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(thresholds, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, n) in enumerate(zip(bars, n_channels)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{n}\nchannels', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 4: Dendrogram showing channel relationships
    ax = fig.add_subplot(gs[2, :])
    linkage_matrix = results['linkage_matrix']
    channel_names = results['channel_names']

    dendrogram = hierarchy.dendrogram(
        linkage_matrix, ax=ax,
        labels=[ch.replace('EEG-', '') for ch in channel_names],
        leaf_rotation=90, leaf_font_size=9, color_threshold=0.3
    )

    ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2,
              label='Aggressive clustering')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
              label='Moderate clustering')
    ax.set_ylabel('Dissimilarity', fontsize=12, fontweight='bold')
    ax.set_title('Channel Clustering Dendrogram\n(Channels merging low are similar)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.suptitle(f'Channel Selection Guide - {subject_id} ({condition})',
                fontsize=17, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def create_cross_subject_recommendations(all_results):
    """Generate channel recommendations across all subjects."""

    # Collect diversity scores for each channel across all subjects and conditions
    channel_scores = {}

    for result in all_results:
        subject = result['subject']
        for condition, cond_data in result['conditions'].items():
            for d in cond_data['diversity_ranking']:
                channel = d['channel']
                score = d['diversity_score']

                if channel not in channel_scores:
                    channel_scores[channel] = []
                channel_scores[channel].append(score)

    # Calculate average and std for each channel
    channel_summary = []
    for channel, scores in channel_scores.items():
        channel_summary.append({
            'channel': channel,
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'consistency': 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
        })

    # Sort by average score
    channel_summary = sorted(channel_summary, key=lambda x: x['avg_score'], reverse=True)

    return channel_summary


def plot_cross_subject_recommendations(channel_summary, save_path):
    """Visualize cross-subject channel recommendations."""

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot 1: Top 15 channels across subjects
    ax = axes[0, 0]
    top_15 = channel_summary[:15]
    channels = [c['channel'].replace('EEG-', '') for c in top_15]
    scores = [c['avg_score'] for c in top_15]
    colors = ['darkgreen' if i < 5 else 'green' if i < 10 else 'yellowgreen'
              for i in range(len(channels))]

    bars = ax.bar(range(len(channels)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Distinctiveness Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Channels Across All Subjects', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Score distribution for top channels
    ax = axes[0, 1]
    top_10 = channel_summary[:10]

    for i, ch_data in enumerate(top_10):
        # Simulate distribution (in real case, you'd store all trial scores)
        mean = ch_data['avg_score']
        std = ch_data['std_score']
        ax.barh(i, mean, xerr=std, alpha=0.7, capsize=5,
                color='steelblue', edgecolor='black')

    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels([c['channel'].replace('EEG-', '') for c in top_10])
    ax.set_xlabel('Distinctiveness Score (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Channels: Consistency', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 3: All channels ranked
    ax = axes[1, 0]
    all_channels = [c['channel'].replace('EEG-', '') for c in channel_summary]
    all_scores = [c['avg_score'] for c in channel_summary]

    colors = ['darkgreen' if i < 5 else 'green' if i < 10 else 'yellowgreen' if i < 15
              else 'orange' if i < 20 else 'red' for i in range(len(all_channels))]

    ax.bar(range(len(all_channels)), all_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=np.mean(all_scores), color='black', linestyle='--', linewidth=2,
              label=f'Average: {np.mean(all_scores):.3f}')
    ax.axvline(x=4.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=9.5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=14.5, color='yellow', linestyle='--', alpha=0.5, linewidth=2)

    ax.set_xlabel('Channel (ranked)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Distinctiveness', fontsize=12, fontweight='bold')
    ax.set_title('All Channels Ranked by Distinctiveness', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(all_channels)))
    ax.set_xticklabels(all_channels, rotation=90, fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Recommended channel sets
    ax = axes[1, 1]
    ax.axis('off')

    recommendations = f"""
{'='*70}
    FINAL CHANNEL SELECTION RECOMMENDATIONS
{'='*70}

TOP 5 CHANNELS (Must Keep - Highest Priority):
    {', '.join([c['channel'] for c in channel_summary[:5]])}

TOP 10 CHANNELS (Recommended - Balanced):
    {', '.join([c['channel'] for c in channel_summary[:10]])}

TOP 15 CHANNELS (Conservative - More Info):
    {', '.join([c['channel'] for c in channel_summary[:15]])}

{'─'*70}

SCORE STATISTICS:
    Top 5 Average:  {np.mean([c['avg_score'] for c in channel_summary[:5]]):.4f}
    Top 10 Average: {np.mean([c['avg_score'] for c in channel_summary[:10]]):.4f}
    Top 15 Average: {np.mean([c['avg_score'] for c in channel_summary[:15]]):.4f}
    Overall Average: {np.mean([c['avg_score'] for c in channel_summary]):.4f}

{'─'*70}

CHANNELS TO REMOVE (Lowest Priority):
    {', '.join([c['channel'] for c in channel_summary[-5:]])}

"""

    ax.text(0.05, 0.95, recommendations, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax.transAxes)

    plt.suptitle('Cross-Subject Channel Selection Recommendations',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def analyze_subject(subject_file):
    # Analyze channel selection for one subject
    print(f"\nAnalyzing {subject_file}...")
    epochs = helpers.load_epochs(subject_file)
    subject_id = subject_file.replace('.fif', '')

    results = {
        'subject': subject_id,
        'conditions': {}
    }

    for condition in epochs.event_id.keys():
        n_trials = len(epochs[condition])
        if n_trials < 2:
            continue

        print(f"  Processing {condition} ({n_trials} trials)...")

        # Identify representative channels
        selection_results = identify_representative_channels(epochs, condition)
        results['conditions'][condition] = selection_results

        # Plot recommendations
        plot_path = output_dir / f'{subject_id}_{condition}_channel_selection.png'
        plot_channel_selection_recommendations(selection_results, subject_id, condition, plot_path)

    return results


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("CHANNEL SELECTION ANALYSIS")
    print("="*80)
    print("\nIdentifying which channels to keep for optimal model performance.\n")

    all_results = []

    for subject in helpers.subjects:
        try:
            result = analyze_subject(f'{subject}.fif')
            all_results.append(result)

        except Exception as e:
            print(f"Error with {subject}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) == 0:
        print("\nERROR: No subjects were successfully analyzed!")
    else:
        # Generate cross-subject recommendations
        print("\n" + "="*80)
        print("GENERATING CROSS-SUBJECT RECOMMENDATIONS")
        print("="*80)

        channel_summary = create_cross_subject_recommendations(all_results)

        # Save summary to file
        summary_path = output_dir / 'final_channel_recommendations.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FINAL CHANNEL RECOMMENDATIONS ACROSS ALL SUBJECTS\n")
            f.write("="*80 + "\n\n")

            f.write("RANKED CHANNELS (by average distinctiveness):\n")
            f.write("-"*80 + "\n")
            for i, ch_data in enumerate(channel_summary, 1):
                f.write(f"{i:3d}. {ch_data['channel']:15s} | "
                       f"Score: {ch_data['avg_score']:.4f} ± {ch_data['std_score']:.4f} | "
                       f"Range: [{ch_data['min_score']:.4f}, {ch_data['max_score']:.4f}]\n")

            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("="*80 + "\n")
            f.write(f"\nTop 5:  {', '.join([c['channel'] for c in channel_summary[:5]])}\n")
            f.write(f"\nTop 10: {', '.join([c['channel'] for c in channel_summary[:10]])}\n")
            f.write(f"\nTop 15: {', '.join([c['channel'] for c in channel_summary[:15]])}\n")
            f.write(f"\nRemove: {', '.join([c['channel'] for c in channel_summary[-5:]])}\n")

        print(f"Saved: {summary_path}")

        # Create visualization
        plot_path = output_dir / 'cross_subject_channel_recommendations.png'
        plot_cross_subject_recommendations(channel_summary, plot_path)
