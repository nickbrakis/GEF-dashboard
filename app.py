from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient
import math
from typing import List, Dict, Optional
from mlflow_metrics import get_eval_metrics, get_timeseries_count
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import io
import os
import tempfile
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Default MLflow tracking URI
DEFAULT_TRACKING_URI = "https://mlflow.gpu.epu.ntua.gr"

# MinIO/S3 Configuration
MINIO_ENDPOINT = "https://api.minio.gpu.epu.ntua.gr"
MINIO_BUCKET = "mlflow-bucket"
# Credentials from environment variables (optional, will try anonymous access first)
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', '')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', '')

@app.route('/')
def index():
    """Serve the main web UI."""
    return send_from_directory('static', 'index.html')

@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    """List all available experiments from MLflow."""
    try:
        tracking_uri = request.args.get('tracking_uri', DEFAULT_TRACKING_URI)
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        experiments = client.search_experiments()
        experiment_list = [
            {
                'name': exp.name,
                'experiment_id': exp.experiment_id,
                'lifecycle_stage': exp.lifecycle_stage
            }
            for exp in experiments
            if exp.lifecycle_stage == 'active'
        ]
        
        return jsonify({
            'success': True,
            'experiments': experiment_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics', methods=['POST'])
def get_metrics():
    """Fetch metrics for a given experiment."""
    try:
        data = request.get_json()
        experiment_name = data.get('experiment_name')
        metrics = data.get('metrics', ['mase'])  # Default to MASE
        tracking_uri = data.get('tracking_uri', DEFAULT_TRACKING_URI)
        parent_filter = data.get('parent_filter')
        
        if not experiment_name:
            return jsonify({
                'success': False,
                'error': 'experiment_name is required'
            }), 400
        
        # Fetch metrics for each requested metric type
        results = {}
        for metric_name in metrics:
            try:
                metric_results = get_eval_metrics(
                    tracking_uri=tracking_uri,
                    experiment_name=experiment_name,
                    metric_name=metric_name,
                    parent_filter=parent_filter,
                    verbose=False
                )
                
                # Fetch timeseries counts and compute weighted average
                valid_runs = []
                metric_values = [] # For min/max/unweighted
                weighted_sum = 0
                total_ts_count = 0
                
                for r in metric_results:
                    val = r["metric_value"]
                    if math.isnan(val):
                        continue # Skip NaNs
                        
                    # Fetch count
                    ts_count = get_timeseries_count(tracking_uri, r["eval_run_id"])
                    
                    if ts_count > 0:
                        r["ts_count"] = ts_count
                        valid_runs.append(r)
                        metric_values.append(val)
                        
                        weighted_sum += val * ts_count
                        total_ts_count += ts_count
                    
                stats = {
                    'total_runs': len(metric_results),
                    'valid_runs': len(valid_runs),
                    'nan_count': len(metric_results) - len(valid_runs), # Approximation
                    'values': metric_values,
                    'runs': valid_runs,
                    'total_timeseries': total_ts_count
                }
                
                if valid_runs and total_ts_count > 0:
                    stats['weighted_average'] = weighted_sum / total_ts_count
                    stats['unweighted_average'] = sum(metric_values) / len(metric_values)
                    stats['average'] = stats['weighted_average'] # Default to weighted for UI compatibility
                    stats['min'] = min(metric_values)
                    stats['max'] = max(metric_values)
                else:
                    stats['weighted_average'] = None
                    stats['unweighted_average'] = None
                    stats['average'] = None
                    stats['min'] = None
                    stats['max'] = None
                
                results[metric_name] = stats
                
            except Exception as e:
                results[metric_name] = {
                    'error': str(e),
                    'total_runs': 0,
                    'valid_runs': 0,
                    'values': []
                }
        
        return jsonify({
            'success': True,
            'experiment_name': experiment_name,
            'metrics': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_experiments():
    """Compare metrics across multiple experiments."""
    try:
        data = request.get_json()
        experiment_names = data.get('experiment_names', [])
        metric_name = data.get('metric', 'mase')  # Single metric for comparison
        tracking_uri = data.get('tracking_uri', DEFAULT_TRACKING_URI)
        parent_filter = data.get('parent_filter')
        
        if not experiment_names or len(experiment_names) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 experiment names are required for comparison'
            }), 400
        
        # Fetch metrics for each experiment
        comparison_results = []
        for exp_name in experiment_names:
            try:
                metric_results = get_eval_metrics(
                    tracking_uri=tracking_uri,
                    experiment_name=exp_name,
                    metric_name=metric_name,
                    parent_filter=parent_filter,
                    verbose=False
                )
                
                # Compute weighted average
                valid_runs = []
                metric_values = []
                weighted_sum = 0
                total_ts_count = 0
                
                for r in metric_results:
                    val = r["metric_value"]
                    if math.isnan(val):
                        continue
                        
                    ts_count = get_timeseries_count(tracking_uri, r["eval_run_id"])
                    
                    if ts_count > 0:
                        metric_values.append(val)
                        weighted_sum += val * ts_count
                        total_ts_count += ts_count
                        valid_runs.append(r)
                
                if valid_runs and total_ts_count > 0:
                    weighted_avg = weighted_sum / total_ts_count
                    unweighted_avg = sum(metric_values) / len(metric_values)
                    min_value = min(metric_values)
                    max_value = max(metric_values)
                else:
                    weighted_avg = None
                    unweighted_avg = None
                    min_value = None
                    max_value = None
                
                comparison_results.append({
                    'experiment_name': exp_name,
                    'metric_name': metric_name,
                    'average': weighted_avg, # Use weighted as main average
                    'weighted_average': weighted_avg,
                    'unweighted_average': unweighted_avg,
                    'min': min_value,
                    'max': max_value,
                    'total_runs': len(metric_results),
                    'valid_runs': len(valid_runs),
                    'total_timeseries': total_ts_count
                })
                
            except Exception as e:
                comparison_results.append({
                    'experiment_name': exp_name,
                    'error': str(e),
                    'average': None,
                    'min': None,
                    'max': None,
                    'total_runs': 0,
                    'valid_runs': 0
                })
        
        return jsonify({
            'success': True,
            'metric_name': metric_name,
            'experiments': comparison_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def fetch_csv_from_mlflow(client: MlflowClient, run_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch evaluation_results_all_ts.csv from MLflow artifacts for a given run.
    """
    try:
        # Based on user description, the path is: eval_results/evaluation_results_all_ts.csv
        artifact_path = "eval_results/evaluation_results_all_ts.csv"
        
        print(f"Attempting to download artifact from run {run_id}: {artifact_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                local_path = client.download_artifacts(run_id, artifact_path, dst_path=tmpdir)
                print(f"Successfully downloaded to: {local_path}")
                df = pd.read_csv(local_path)
                print(f"CSV loaded with {len(df)} rows")
                return df
            except Exception as e:
                print(f"Failed to download direct path: {str(e)}")
                return None
    except Exception as e:
        print(f"Error fetching CSV from MLflow: {str(e)}")
        return None

def get_experiment_data_df(experiment_name: str, tracking_uri: str, parent_filter: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Reusable function to fetch and concatenate evaluation results from an experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment {experiment_name} not found")
        return None
    
    experiment_id = experiment.experiment_id
    
    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1000
    )
    
    if not all_runs:
        return None
        
    # Build a map for easy lookup of run names
    run_id_to_name = {run.info.run_id: run.data.tags.get("mlflow.runName", "Unknown") for run in all_runs}
    
    # Identify parent runs
    parent_run_ids = set()
    for run in all_runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if not parent_id: # Root run
            # If a filter is provided, check if the run name matches 
            # (assuming filter is the run name for now as per user request)
            run_name = run.data.tags.get("mlflow.runName", "")
            if parent_filter:
                if parent_filter == run_name:
                    parent_run_ids.add(run.info.run_id)
            else:
                 parent_run_ids.add(run.info.run_id)

    all_dataframes = []
    
    for run in all_runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        child_run_name = run.data.tags.get("mlflow.runName", "")
        
        # Check if this is an eval child run
        if parent_id and parent_id in parent_run_ids and "eval" in child_run_name.lower():
            parent_run_name = run_id_to_name.get(parent_id, "Unknown")
            
            try:
                df = fetch_csv_from_mlflow(client, run.info.run_id)
                if df is not None and not df.empty:
                    df.insert(0, 'parent_run_name', parent_run_name)
                    all_dataframes.append(df)
            except Exception as e:
                print(f"Error processing run {run.info.run_id}: {e}")
                
    if not all_dataframes:
        return None
    
    return pd.concat(all_dataframes, ignore_index=True)

@app.route('/api/plot-comparison', methods=['POST'])
def plot_comparison_endpoint():
    """
    Generate and return a comparison plot between two experiments (Global vs Grouped).
    """
    try:
        data = request.get_json()
        global_experiment = data.get('global_experiment')
        grouped_experiment = data.get('grouped_experiment')
        global_parent_filter = data.get('global_parent_filter') # New optional parameter
        tracking_uri = data.get('tracking_uri', DEFAULT_TRACKING_URI)
        
        if not global_experiment or not grouped_experiment:
            return jsonify({'success': False, 'error': 'Both experiment names are required'}), 400

        # Fetch data
        # Apply filter only to global experiment as requested
        df_global = get_experiment_data_df(global_experiment, tracking_uri, parent_filter=global_parent_filter)
        df_grouped = get_experiment_data_df(grouped_experiment, tracking_uri)
        
        if df_global is None or df_global.empty:
            return jsonify({'success': False, 'error': f'No data found for {global_experiment}'}), 404
        if df_grouped is None or df_grouped.empty:
            return jsonify({'success': False, 'error': f'No data found for {grouped_experiment}'}), 404
            
        # Data Processing (Logic from plot_compare.py)
        # --------------------------------------------
        
        # 1. Map Timeseries ID to Groups from the Grouped DF
        # We assume the grouped structure is in df_grouped
        ts_mapping = df_grouped[['Timeseries ID', 'parent_run_name']].drop_duplicates()
        ts_mapping = ts_mapping.drop_duplicates(subset=['Timeseries ID'])
        
        # Determine Grouping Name
        try:
            example_curr = ts_mapping['parent_run_name'].iloc[0]
            parts = example_curr.split('_')
            # Expecting format: <model>_<GROUP>_<group id>
            if len(parts) >= 3:
                group_type = parts[1]
            else:
                group_type = "Grouped"
        except Exception:
            group_type = "Grouped"
            
        group_label = group_type.capitalize()
        grouped_mase_col = f'{group_label}_MASE'
        
        # 2. Prepare Global Data with Groups
        # Inner join to restrict Global results to only those TS present in the Grouped experiment
        # We drop parent_run_name from df_global to avoid collision/suffixes, as we want the grouping from ts_mapping
        df_global_subset = df_global[['Timeseries ID', 'mase']]
        df_global_mapped = df_global_subset.merge(ts_mapping, on='Timeseries ID', how='inner')
        
        # 3. Rename columns
        df_global_mapped = df_global_mapped[['Timeseries ID', 'parent_run_name', 'mase']].rename(columns={'mase': 'Global_MASE'})
        df_grouped_clean = df_grouped[['Timeseries ID', 'mase']].rename(columns={'mase': grouped_mase_col})
        
        # 4. Merge
        # Check if we have common Timeseries IDs
        if pd.merge(df_global_mapped, df_grouped_clean, on='Timeseries ID', how='inner').empty:
             return jsonify({'success': False, 'error': 'No matching Timeseries IDs found between the two experiments'}), 400
             
        df_merged = df_global_mapped.merge(df_grouped_clean, on='Timeseries ID', how='inner')
        
        # 5. Calculate Diff
        df_merged['Diff'] = df_merged['Global_MASE'] - df_merged[grouped_mase_col]
        df_merged = df_merged.sort_values(by=['parent_run_name', 'Diff'])
        
        # Plotting
        sns.set_theme(style="whitegrid")
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot 1: Aggregated
        group_agg = df_merged.groupby('parent_run_name')[['Global_MASE', grouped_mase_col]].mean().reset_index()
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
        axes[0].tick_params(axis='x', rotation=45)
        
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.3f', padding=3)
            
        # Plot 2: Individual
        colors = ['#4c72b0' if x < 0 else '#c44e52' for x in df_merged['Diff']]
        x_positions = range(len(df_merged))
        
        axes[1].bar(x_positions, df_merged['Diff'], color=colors, alpha=0.8)
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set_title(f'Difference in MASE per Time Series (Global - {group_label})', fontsize=16, pad=10)
        axes[1].set_ylabel('MASE Difference', fontsize=12)
        axes[1].set_xlabel(f'Time Series (Grouped by {group_label})', fontsize=12)
        
        axes[1].text(0.02, 0.95, 'Bars below 0: Global Model is Better', transform=axes[1].transAxes, 
                     color='#4c72b0', fontweight='bold', fontsize=12)
        axes[1].text(0.02, 0.91, f'Bars above 0: {group_label} Model is Better', transform=axes[1].transAxes, 
                     color='#c44e52', fontweight='bold', fontsize=12)
                     
        # Group labels on x-axis (bottom plot)
        group_counts = df_merged['parent_run_name'].value_counts(sort=False).reindex(df_merged['parent_run_name'].unique())
        current_pos = 0
        ylim = axes[1].get_ylim()
        y_pos = ylim[0] - (ylim[1]-ylim[0])*0.05
        
        for group, count in group_counts.items():
            center = current_pos + count / 2
            axes[1].text(center, y_pos, group, ha='center', va='top', fontweight='bold', rotation=90 if len(group)>10 else 0)
            if current_pos > 0:
                axes[1].axvline(current_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
            current_pos += count
            
        axes[1].set_xticks([])
        
        plt.tight_layout()
        
        # Save to buffer
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close(fig)
        
        return Response(img_buf, mimetype='image/png')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-csv', methods=['POST'])
def export_concatenated_csv():
    """
    Concatenate all evaluation CSVs from an experiment and return with summary metrics.
    """
    try:
        data = request.get_json()
        experiment_name = data.get('experiment_name')
        parent_filter = data.get('parent_filter')
        tracking_uri = data.get('tracking_uri', DEFAULT_TRACKING_URI)
        
        if not experiment_name:
            return jsonify({
                'success': False,
                'error': 'experiment_name is required'
            }), 400
        
        # Use the helper function
        final_df = get_experiment_data_df(experiment_name, tracking_uri, parent_filter)
        
        if final_df is None or final_df.empty:
            return jsonify({
                'success': False,
                'error': 'No CSV files found for this experiment'
            }), 404
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Create response
        response = Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={experiment_name}_concatenated.csv'
            }
        )
        
        return response
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'default_tracking_uri': DEFAULT_TRACKING_URI
    })

if __name__ == '__main__':
    print("=" * 60)
    print("MLflow Metrics Visualization App")
    print("=" * 60)
    print(f"Server starting at: http://localhost:5000")
    print(f"Default MLflow URI: {DEFAULT_TRACKING_URI}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
