from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient
import math
from typing import List, Dict, Optional
from mlflow_metrics import get_eval_metrics, get_timeseries_count

app = Flask(__name__, static_folder='static')
CORS(app)

# Default MLflow tracking URI
DEFAULT_TRACKING_URI = "https://mlflow.gpu.epu.ntua.gr"

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
