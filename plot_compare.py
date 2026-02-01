import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison():
    # Files
    global_csv_path = 'mlp_global_evaluation_results_all_ts.csv'
    grouped_csv_path = 'MLP_counties_concatenated_2026-02-01.csv'
    output_image = 'model_comparison_plot.png'

    # Load Data
    try:
        df_global = pd.read_csv(global_csv_path)
        df_grouped = pd.read_csv(grouped_csv_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # Data Processing
    # ----------------
    # 1. Map Timeseries ID to Groups (parent_run_name)
    # Generic name: df_grouped
    ts_mapping = df_grouped[['Timeseries ID', 'parent_run_name']].drop_duplicates()
    
    # Handle duplicates if any (keep first)
    ts_mapping = ts_mapping.drop_duplicates(subset=['Timeseries ID'])

    # Determine Grouping Name from data EARLY to use in column names
    # User specified format: <model>_<GROUP>_<group id>
    try:
        example_curr = ts_mapping['parent_run_name'].iloc[0]
        parts = example_curr.split('_')
        if len(parts) >= 3:
            group_type = parts[1] # e.g., 'pt', 'counties', 'store'
        else:
            group_type = "Grouped"
    except Exception:
        group_type = "Grouped"
        
    group_label = group_type.capitalize() # e.g. "Counties", "Pt"
    grouped_mase_col = f'{group_label}_MASE'

    # 2. Prepare Global Data with Groups
    df_global_mapped = df_global.merge(ts_mapping, on='Timeseries ID', how='inner')
    
    # 3. Rename columns for clarity before merge
    df_global_mapped = df_global_mapped[['Timeseries ID', 'parent_run_name', 'mase']].rename(columns={'mase': 'Global_MASE'})
    # Rename generic grouped dataframe column dynamically
    df_grouped_clean = df_grouped[['Timeseries ID', 'mase']].rename(columns={'mase': grouped_mase_col})

    # 4. Merge Global and Grouped data at Timeseries level
    df_merged = df_global_mapped.merge(df_grouped_clean, on='Timeseries ID', how='inner')
    
    # 5. Calculate Difference
    # Negative Diff means Global < Grouped (Global is better)
    df_merged['Diff'] = df_merged['Global_MASE'] - df_merged[grouped_mase_col]
    
    # Sort for better visualization (by Group then by Diff)
    df_merged = df_merged.sort_values(by=['parent_run_name', 'Diff'])

    # Plotting
    # --------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2]})

    # PLOT 1: Aggregated Group Performance (Grouped Bar Chart)
    # ------------------------------------------------------
    # Aggregate data
    group_agg = df_merged.groupby('parent_run_name')[['Global_MASE', grouped_mase_col]].mean().reset_index()
    # Melt for seaborn
    group_melt = group_agg.melt(id_vars='parent_run_name', var_name='Model', value_name='Average MASE')
    
    sns.barplot(
        data=group_melt, 
        x='parent_run_name', 
        y='Average MASE', 
        hue='Model', 
        palette={'Global_MASE': '#4c72b0', grouped_mase_col: '#c44e52'},
        ax=axes[0]
    )
    
    axes[0].set_title(f'Average MASE by {group_label} (Lower is Better)', fontsize=16, pad=10)
    axes[0].set_xlabel(f'{group_label} Group', fontsize=12)
    axes[0].set_ylabel('MASE', fontsize=12)
    axes[0].legend(title='Model')
    
    # Rotate x-axis labels for better readability
    axes[0].tick_params(axis='x', rotation=45)

    # Add value labels
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.3f', padding=3)


    # PLOT 2: Individual Timeseries Diff (Diverging Bar Chart)
    # ------------------------------------------------------
    # Color bars based on who wins
    # Global wins (Diff < 0) -> Blue, Grouped wins (Diff > 0) -> Red
    colors = ['#4c72b0' if x < 0 else '#c44e52' for x in df_merged['Diff']]
    
    # Create the interaction grouping for x-axis labeling if complex, 
    # but here we just plot bar by bar.
    x_positions = range(len(df_merged))
    
    axes[1].bar(x_positions, df_merged['Diff'], color=colors, alpha=0.8)
    
    # Aesthetic improvements
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title(f'Difference in MASE per Time Series (Global - {group_label})', fontsize=16, pad=10)
    axes[1].set_ylabel('MASE Difference', fontsize=12)
    axes[1].set_xlabel(f'Time Series (Grouped by {group_label})', fontsize=12)
    
    # Add text annotation
    axes[1].text(0.02, 0.95, 'Bars below 0: Global Model is Better', transform=axes[1].transAxes, 
                 color='#4c72b0', fontweight='bold', fontsize=12)
    axes[1].text(0.02, 0.91, f'Bars above 0: {group_label} Model is Better', transform=axes[1].transAxes, 
                 color='#c44e52', fontweight='bold', fontsize=12)

    # Label groups on x-axis (approximate positions)
    # Find transitions between groups
    group_counts = df_merged['parent_run_name'].value_counts(sort=False).reindex(df_merged['parent_run_name'].unique())
    current_pos = 0
    for group, count in group_counts.items():
        center = current_pos + count / 2
        axes[1].text(center, axes[1].get_ylim()[0] - (axes[1].get_ylim()[1]-axes[1].get_ylim()[0])*0.05, 
                     group, ha='center', va='top', fontweight='bold')
        # Add separator line
        if current_pos > 0:
            axes[1].axvline(current_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
        current_pos += count

    # Remove x ticks as individual IDs are too many/cluttered
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    plot_comparison()
