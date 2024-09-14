import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot histograms for markers
def plot_marker_histograms(df, markers):
    fig, axes = plt.subplots(len(markers), 1, figsize=(10, 5 * len(markers)))
    if len(markers) == 1:  # Adjust if only one marker is provided
        axes = [axes]
    for ax, marker in zip(axes, markers):
        sns.histplot(df[marker], bins=100, kde=True, ax=ax)
        ax.set_title(f'Histogram of {marker}')
        ax.set_ylim(0, 50)
        
        ax.set_xlabel('Marker Value')
        ax.set_ylabel('Number of Cells')
    plt.tight_layout()
    plt.show()

# Scatter plot 
# args: df, subset_index, thresholds={'CD3': 0.5, 'CD45': 1}
def plot_t_cell_subtypes(df, index):
    
    filtered = df[index]  # Filter CD3+ cells
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered['CD14'], filtered['CD68'], c='blue', s=0.2)
    # plt.axvline(x=cd4_threshold, color='red', linestyle='--', label='CD4 threshold')
    # plt.axhline(y=cd8_threshold, color='green', linestyle='--', label='CD8 threshold')
    # plt.xlabel('CD14 Expression Level')
    # plt.ylabel('CD68 Expression Level')
    # plt.title('Scatter Plot of T-cell Subtypes based on CD14 and CD68 Expression')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Concatenate matched pairs text files.")

    parser.add_argument(
        "--input_file", required=True, 
        help="The file containing the matched pairs.")
    
    args = parser.parse_args()

    df = pd.read_table(args.input_file, sep='\t')
    # markers = ['CD45']
    # plot_marker_histograms(df, markers)

    cd3_neg_index = (df['CD45'] > 1.0) & ~(df['CD3'] > 0.5) 
    cd20_neg_index = (df['CD45'] > 1.0) & ~(df['CD3'] > 0.5) & ~(df['CD20'] > 0.5)
    cd56_neg_index = (df['CD45'] > 1.0) & ~(df['CD3'] > 0.5) & ~(df['CD20'] > 0.5) & ~(df['CD56'] > 0.5)

    plot_t_cell_subtypes(df, cd56_neg_index)

    filtered_cell = df[cd20_neg_index]
    markers = ['CD56']
    plot_marker_histograms(filtered_cell, markers)

if __name__ == "__main__":
    main()