import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_core_classes(json_path, output_path):
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Calculate number of classes per document
    class_counts = [len(classes) for classes in data.values()]
    
    # Filter out empty documents for specific plots
    non_empty_counts = [c for c in class_counts if c > 0]

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of Level 0 Classes per Document (Bar Chart)
    # Count occurrences of each length
    count_dist = Counter(class_counts)
    max_count = max(class_counts) if class_counts else 0
    x_values = range(max_count + 1)
    y_values = [count_dist.get(x, 0) for x in x_values]
    
    # Limit x-axis if there are too many unique counts, or group 6+
    # The example image shows 0-6. Let's see what our max is.
    # If max is large, we might want to cap it at 6 or 7 for the bar chart like the example.
    
    display_x = list(range(7))
    display_y = [count_dist.get(x, 0) for x in display_x]
    # Add 6+ if needed, but the image shows just 6. Let's stick to the data.
    # If the data has > 6, we should probably group them or show them.
    # The image shows "6", implying exactly 6. 
    # Let's just plot all present counts for now, but if it's too wide, we'll truncate.
    
    axes[0, 0].bar(x_values, y_values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Level 0 Classes per Document')
    axes[0, 0].set_xlabel('Number of Level 0 Classes')
    axes[0, 0].set_ylabel('Number of Documents')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_xticks(x_values)

    # 2. Distribution (Excluding Empty Documents) (Pie Chart)
    if non_empty_counts:
        pie_counts = Counter(non_empty_counts)
        # Group small percentages if necessary? The image shows 1, 2, 3, 4, 5, 6.
        labels = [f'{k} classes' for k in pie_counts.keys()]
        sizes = pie_counts.values()
        
        # Sort by key to make it ordered
        sorted_pie = sorted(pie_counts.items())
        labels = [f'{k} classes' for k, v in sorted_pie]
        sizes = [v for k, v in sorted_pie]
        
        # Use a nice color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        axes[0, 1].set_title('Distribution (Excluding Empty Documents)')
    else:
        axes[0, 1].text(0.5, 0.5, 'No non-empty documents', ha='center')

    # 3. Cumulative Distribution (Line Chart)
    # Calculate cumulative percentages
    sorted_counts = sorted(class_counts)
    n = len(sorted_counts)
    if n > 0:
        # We want the percentage of documents having <= k classes
        # x_values are the unique counts (0, 1, 2...)
        unique_counts = sorted(list(set(class_counts)))
        cumulative_y = []
        for k in unique_counts:
            count_le_k = sum(1 for c in class_counts if c <= k)
            cumulative_y.append(count_le_k / n * 100)
            
        axes[1, 0].plot(unique_counts, cumulative_y, marker='o', color='green', linewidth=2)
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].set_xlabel('Number of Level 0 Classes')
        axes[1, 0].set_ylabel('Cumulative Percentage (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add 50% and 90% lines
        axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
        axes[1, 0].axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
        axes[1, 0].legend()
        axes[1, 0].set_xticks(unique_counts)

    # 4. Histogram (Excluding Empty Documents)
    if non_empty_counts:
        mean_val = np.mean(non_empty_counts)
        median_val = np.median(non_empty_counts)
        
        # Bins centered on integers
        bins = np.arange(min(non_empty_counts) - 0.5, max(non_empty_counts) + 1.5, 1)
        
        axes[1, 1].hist(non_empty_counts, bins=bins, color='lightsalmon', edgecolor='black', alpha=0.8)
        axes[1, 1].set_title('Histogram (Excluding Empty Documents)')
        axes[1, 1].set_xlabel('Number of Level 0 Classes')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add mean and median lines
        axes[1, 1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[1, 1].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val}')
        axes[1, 1].legend()
        
        # Set x-ticks to integers
        axes[1, 1].set_xticks(range(min(non_empty_counts), max(non_empty_counts) + 1))

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")

if __name__ == "__main__":
    json_path = 'checkpoints/core_classes.json'
    output_path = 'core_classes_analysis.png'
    analyze_core_classes(json_path, output_path)