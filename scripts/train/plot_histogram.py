import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(data, ylabel, title, save_name):
    datasets = list(data.keys())
    fusion_methods = ['series', 'parallel', 'cross']

    # Extract scores for each fusion method
    series_scores = [data[dataset]['series'] for dataset in datasets]
    parallel_scores = [data[dataset]['parallel'] for dataset in datasets]
    cross_scores = [data[dataset]['cross'] for dataset in datasets]

    bar_width = 0.25
    positions_series = np.arange(len(datasets))
    positions_parallel = [x + bar_width for x in positions_series]
    positions_cross = [x + bar_width for x in positions_parallel]

    # Subtle colors for research paper
    colors = ['#AEC7E8', '#FFBB78', '#98DF8A']

    # Check if the label is already in the legend
    if 'Series' not in plt.gca().get_legend_handles_labels()[1]:
        plt.bar(positions_series, series_scores, color=colors[0], width=bar_width, edgecolor='grey', label='Series')
    else:
        plt.bar(positions_series, series_scores, color=colors[0], width=bar_width, edgecolor='grey')

    if 'Parallel' not in plt.gca().get_legend_handles_labels()[1]:
        plt.bar(positions_parallel, parallel_scores, color=colors[1], width=bar_width, edgecolor='grey', label='Parallel')
    else:
        plt.bar(positions_parallel, parallel_scores, color=colors[1], width=bar_width, edgecolor='grey')

    if 'Cross' not in plt.gca().get_legend_handles_labels()[1]:
        plt.bar(positions_cross, cross_scores, color=colors[2], width=bar_width, edgecolor='grey', label='Cross')
    else:
        plt.bar(positions_cross, cross_scores, color=colors[2], width=bar_width, edgecolor='grey')

    plt.xlabel('Datasets', fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(datasets))], datasets)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))  # Adjust legend position to avoid overlap
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

# NetSurfP-2.0, CB513, CASP12, TS115

f1_scores = {
    'NetSurfP-2.0': {'series': 0.6140, 'parallel': 0.6257, 'cross': 0.57112},
    'CASP12': {'series': 0.5048, 'parallel': 0.5004, 'cross': 0.4118},
    'CB513': {'series': 0.5675, 'parallel': 0.569, 'cross': 0.5370},
    'TS115': {'series': 0.5979, 'parallel': 0.5964, 'cross': 0.5606},
}
accuracy = {
    'NetSurfP-2.0': {'series': 0.6962, 'parallel': 0.7023, 'cross': 0.6992},
    'CASP12': {'series': 0.5844, 'parallel': 0.5861, 'cross': 0.5804},
    'CB513': {'series':  0.6568, 'parallel': 0.6567 , 'cross': 0.6452},
    'TS115': {'series': 0.7063, 'parallel': 0.7033, 'cross': 0.6936},
}
loss = {
    'NetSurfP-2.0': {'series': 0.8224, 'parallel': 0.8964, 'cross': 0.8385},
    'CASP12': {'series': 1.1280, 'parallel': 1.1323, 'cross': 1.1046},
    'CB513': {'series': 0.9413, 'parallel': 0.9577, 'cross': 0.9471},
    'TS115': {'series': 0.8283, 'parallel': 0.839, 'cross': 0.8372},
}

# Plotting
plot_comparison(f1_scores, 'F1 Score', 'Comparison of F1 Scores for Different Fusion Methods', 'f1_scores.png')
plot_comparison(accuracy, 'Accuracy', 'Comparison of Accuracy Scores for Different Fusion Methods', 'Accuracy.png')
plot_comparison(loss, 'Loss', 'Comparison of Loss for Different Fusion Methods', 'Loss.png')
