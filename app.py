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
        
        # Set up MLflow client
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return jsonify({
                'success': False,
                'error': f'Experiment "{experiment_name}" not found'
            }), 404
        
        experiment_id = experiment.experiment_id
        
        # Get all runs for the experiment in one go
        all_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="", # Fetch everything, we'll filter in-memory
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=1000
        )
        
        if not all_runs:
            return jsonify({
                'success': False,
                'error': 'No runs found for this experiment'
            }), 404
            
        # Build a map for easy lookup of run names
        run_id_to_name = {run.info.run_id: run.data.tags.get("mlflow.runName", "Unknown") for run in all_runs}
        
        # Map to track parent runs that match the user's filter (if provided)
        # If no filter, all root runs are parents
        parent_run_ids = set()
        for run in all_runs:
            parent_id = run.data.tags.get("mlflow.parentRunId")
            if not parent_id: # Root run
                # Apply user filter manually if provided (simpler to just check if it was in parent_runs before)
                # But here we'll just check if it matches the name for now or just include all
                parent_run_ids.add(run.info.run_id)

        # Collect all CSVs from eval child runs
        all_dataframes = []
        successful_runs = 0
        failed_runs = 0
        
        print(f"\n=== Processing experiment: {experiment_name} ===")
        print(f"Total runs found: {len(all_runs)}")
        
        for run in all_runs:
            parent_id = run.data.tags.get("mlflow.parentRunId")
            child_run_name = run.data.tags.get("mlflow.runName", "")
            
            # Check if this is an eval child run of one of our parents
            if parent_id and parent_id in parent_run_ids and "eval" in child_run_name.lower():
                parent_run_name = run_id_to_name.get(parent_id, "Unknown")
                print(f"  Eval run found: {child_run_name} (Parent: {parent_run_name})")
                
                try:
                    df = fetch_csv_from_mlflow(client, run.info.run_id)
                    if df is not None and not df.empty:
                        # Add metadata columns
                        df.insert(0, 'parent_run_name', parent_run_name)
                        all_dataframes.append(df)
                        successful_runs += 1
                        print(f"    -> SUCCESS: Added {len(df)} rows")
                    else:
                        failed_runs += 1
                        print(f"    -> FAILED: No data or artifact missing")
                except Exception as e:
                    failed_runs += 1
                    print(f"    -> ERROR: {str(e)}")
        
        print(f"\n=== Summary ===")
        print(f"Successful: {successful_runs}, Failed: {failed_runs}")
        print(f"Total DataFrames collected: {len(all_dataframes)}")
        
        
        if not all_dataframes:
            return jsonify({
                'success': False,
                'error': 'No CSV files found for this experiment'
            }), 404
        
        # Concatenate all DataFrames
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
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
