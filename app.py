from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient
import math
from typing import List, Dict, Optional
from mlflow_metrics import get_eval_metrics

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
                
                # Process the results and filter out NaN values
                metric_values = [r["metric_value"] for r in metric_results]
                valid_values = [v for v in metric_values if not math.isnan(v)]
                
                # Filter runs to only include those with valid (non-NaN) metric values
                valid_runs = [
                    r for r in metric_results 
                    if not math.isnan(r["metric_value"])
                ]
                
                stats = {
                    'total_runs': len(metric_results),
                    'valid_runs': len(valid_values),
                    'nan_count': len(metric_values) - len(valid_values),
                    'values': valid_values,
                    'runs': valid_runs  # Only include runs with valid values
                }
                
                if valid_values:
                    stats['average'] = sum(valid_values) / len(valid_values)
                    stats['min'] = min(valid_values)
                    stats['max'] = max(valid_values)
                else:
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
