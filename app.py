from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import mlflow
from mlflow.tracking import MlflowClient
import math
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import requests
from requests.exceptions import RequestException
import json
import time
import requests
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Default MLflow tracking URI
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.gpu.epu.ntua.gr")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "http://127.0.0.1:7001")
MCP_TIMEOUT_SECONDS = int(os.environ.get("MCP_TIMEOUT_SECONDS", "45"))
_mcp_next_id = 1


def _mcp_request(method: str, params: Dict, timeout_seconds: int = MCP_TIMEOUT_SECONDS) -> Dict:
    global _mcp_next_id
    req_id = _mcp_next_id
    _mcp_next_id += 1
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    response = requests.post(
        f"{MCP_BASE_URL}/mcp",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()

# MinIO/S3 Configuration
MINIO_ENDPOINT = "https://api.minio.gpu.epu.ntua.gr"
MINIO_BUCKET = "mlflow-bucket"
# Credentials from environment variables (optional, will try anonymous access first)
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', '')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', '')

# Leaderboard caching (simple in-memory cache)
LEADERBOARD_CACHE_TTL_SECONDS = int(os.environ.get("LEADERBOARD_CACHE_TTL_SECONDS", "900"))
_leaderboard_cache = {"timestamp": None, "payload": None, "tracking_uri": None}

_timeseries_count_cache: Dict[tuple, int] = {}


def get_eval_metrics(
    tracking_uri: str,
    experiment_name: str,
    metric_name: str = "mase",
    parent_filter: Optional[str] = None,
    verbose: bool = False,
) -> List[Dict]:
    """
    Retrieve metrics from eval subruns of an MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    if verbose:
        print(f"Found experiment: {experiment_name} (ID: {experiment_id})")

    parent_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1000,
    )

    if parent_filter:
        parent_runs = [r for r in parent_runs if r.data.tags.get("mlflow.runName") == parent_filter]

    if verbose:
        print(f"Found {len(parent_runs)} runs")

    results = []
    for parent_run in parent_runs:
        parent_run_name = parent_run.data.tags.get("mlflow.runName", "Unknown")

        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=1000,
        )

        for child_run in child_runs:
            child_run_name = child_run.data.tags.get("mlflow.runName", "")

            if "eval" in child_run_name.lower():
                metrics = child_run.data.metrics
                if metric_name in metrics:
                    result = {
                        "parent_run_id": parent_run.info.run_id,
                        "parent_run_name": parent_run_name,
                        "eval_run_id": child_run.info.run_id,
                        "eval_run_name": child_run_name,
                        "metric_name": metric_name,
                        "metric_value": metrics[metric_name],
                    }
                    results.append(result)

                    if verbose:
                        print(f"  Parent: {parent_run_name}")
                        print(f"    Eval: {child_run_name}")
                        print(f"    {metric_name}: {metrics[metric_name]}")
    return results


def get_timeseries_count(tracking_uri: str, run_id: str) -> int:
    """
    Get the number of timeseries in a run by counting directories in 'eval_results'
    using the MLflow REST API.
    """
    cache_key = (tracking_uri, run_id)
    cached_value = _timeseries_count_cache.get(cache_key)
    if cached_value is not None:
        return cached_value

    try:
        base_uri = tracking_uri.rstrip('/')
        url = f"{base_uri}/api/2.0/mlflow/artifacts/list"

        params = {
            "run_id": run_id,
            "path": "eval_results",
        }

        timeouts = [(5, 20), (5, 30), (10, 45)]
        last_error = None
        for timeout in timeouts:
            try:
                response = requests.get(url, params=params, timeout=timeout)
                if response.status_code == 200:
                    data = response.json()
                    files = data.get("files", [])
                    directories = [f for f in files if f.get("is_dir")]
                    count = len(directories)
                    _timeseries_count_cache[cache_key] = count
                    return count
                if response.status_code in {404, 403}:
                    _timeseries_count_cache[cache_key] = 0
                    return 0
                last_error = f"status {response.status_code}"
            except RequestException as e:
                last_error = e
                continue
        print(f"  ⚠️  Error counting timeseries for run {run_id}: {last_error}")
        return 0

    except Exception as e:
        print(f"  ⚠️  Error counting timeseries for run {run_id}: {e}")
        return 0


def get_timeseries_counts(tracking_uri: str, run_ids: List[str], max_workers: int = 12) -> Dict[str, int]:
    """
    Fetch timeseries counts for a list of run_ids in parallel.
    Uses cached get_timeseries_count to avoid repeated API calls.
    """
    unique_run_ids = {run_id for run_id in run_ids if run_id}
    if not unique_run_ids:
        return {}

    counts: Dict[str, int] = {}
    worker_count = min(max_workers, len(unique_run_ids))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(get_timeseries_count, tracking_uri, run_id): run_id
            for run_id in unique_run_ids
        }
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                counts[run_id] = future.result()
            except Exception:
                counts[run_id] = 0
    return counts


def compute_weighted_metric_stats(results: List[Dict], counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    """
    Compute weighted statistics from metric results using pre-fetched counts.
    """
    metric_values = []
    weighted_sum = 0.0
    total_ts_count = 0

    for r in results:
        val = r["metric_value"]
        if math.isnan(val):
            continue
        ts_count = counts.get(r["eval_run_id"], 0)
        if ts_count > 0:
            metric_values.append(val)
            weighted_sum += val * ts_count
            total_ts_count += ts_count

    if metric_values and total_ts_count > 0:
        return {
            "weighted_average": weighted_sum / total_ts_count,
            "unweighted_average": sum(metric_values) / len(metric_values),
            "min": min(metric_values),
            "max": max(metric_values),
            "total_timeseries": total_ts_count,
        }

    return {
        "weighted_average": None,
        "unweighted_average": None,
        "min": None,
        "max": None,
        "total_timeseries": 0,
    }

@app.route('/')
def index():
    """Serve the main web UI."""
    return send_from_directory('static', 'index.html')


def _mcp_call_tool(tool_name: str, arguments: Dict, timeout_seconds: int = 45) -> Dict:
    if not arguments.get("tracking_uri"):
        arguments["tracking_uri"] = DEFAULT_TRACKING_URI

    tool_map = {
        "mlflow_list_experiments": "mlflow.list_experiments",
        "mlflow_list_runs": "mlflow.list_runs",
        "mlflow_get_run": "mlflow.get_run",
        "mlflow_list_pipeline_runs": "mlflow.list_pipeline_runs",
        "mlflow_get_experiment_evaluation": "mlflow.get_experiment_evaluation",
    }
    mcp_tool = tool_map.get(tool_name)
    if not mcp_tool:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_response = _mcp_request(
        "tools/call",
        {"name": mcp_tool, "arguments": arguments},
        timeout_seconds=timeout_seconds,
    )

    if not tool_response:
        raise RuntimeError("No tool response received from MCP server")

    if "error" in tool_response:
        detail = tool_response["error"].get("data", {}).get("detail")
        message = tool_response["error"].get("message", "MCP tool error")
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)

    content = tool_response.get("result", {}).get("content", [])
    if not content:
        return {}
    text_payload = content[0].get("text", "{}")
    return json.loads(text_payload)


def _chat_llm(message: str, tracking_uri: str) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    tools = [
        {
            "type": "function",
            "name": "mlflow_list_experiments",
            "description": "List active MLflow experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string"},
                },
            },
        },
        {
            "type": "function",
            "name": "mlflow_list_runs",
            "description": "List runs for an experiment by name or id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string"},
                    "experiment_name": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "max_results": {"type": "integer", "default": 50},
                    "filter_string": {"type": "string"},
                    "order_by": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "name": "mlflow_get_run",
            "description": "Get details for a run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string"},
                    "run_id": {"type": "string"},
                },
                "required": ["run_id"],
            },
        },
        {
            "type": "function",
            "name": "mlflow_list_pipeline_runs",
            "description": "List parent pipeline runs and their child stages (load_data, etl, train_<model>, eval).",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string"},
                    "experiment_name": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "max_results": {"type": "integer", "default": 50},
                    "include_children": {"type": "boolean", "default": True},
                    "parent_name_contains": {"type": "string"},
                },
            },
        },
        {
            "type": "function",
            "name": "mlflow_get_experiment_evaluation",
            "description": (
                "Compute experiment-level eval metrics from eval child runs. "
                "In aggregate_mode='auto', experiments with prefix 'GEF' are unweighted, "
                "all others are weighted by timeseries counts inferred from eval artifacts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string"},
                    "experiment_name": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "aggregate_mode": {
                        "type": "string",
                        "enum": ["auto", "weighted", "unweighted"],
                        "default": "auto",
                    },
                    "parent_run_name": {"type": "string"},
                    "include_parent_breakdown": {"type": "boolean", "default": False},
                },
            },
        },
    ]

    input_list = [
        {
            "role": "system",
            "content": (
                "You are an MLflow assistant for time-series forecasting pipelines. "
                "Use tools to fetch factual data before answering. "
                "Pipeline structure: parent run is a full job; children are load_data, etl, train_<Model>, eval. "
                "For total experiment evaluation, use mlflow_get_experiment_evaluation with aggregate_mode='auto' "
                "unless user explicitly requests a different aggregation. "
                "Summarize results clearly and concisely, and call out whether weighted or unweighted aggregation was used."
                "Don't provide more metrics than mase, except it is asked for."
            ),
        },
        {"role": "user", "content": message},
    ]

    for _ in range(6):
        response = client.responses.create(
            model=OPENAI_MODEL,
            tools=tools,
            input=input_list,
        )
        input_list += response.output
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            return response.output_text

        for item in tool_calls:
            args = json.loads(item.arguments or "{}")
            args["tracking_uri"] = tracking_uri
            tool_output = _mcp_call_tool(item.name, args)
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(tool_output),
                }
            )

    return "I could not complete the tool-calling workflow within the allowed number of steps."

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

@app.route('/api/chat', methods=['POST'])
def chat_query():
    """
    Minimal chat endpoint for natural language queries about MLflow experiments.
    """
    try:
        data = request.get_json()
        message = (data.get('message') or '').strip()
        tracking_uri = data.get('tracking_uri', DEFAULT_TRACKING_URI)

        if not message:
            return jsonify({'success': False, 'error': 'message is required'}), 400

        if not OPENAI_API_KEY:
            return jsonify({
                'success': False,
                'error': 'OPENAI_API_KEY is not set on the server'
            }), 500

        response_text = _chat_llm(message, tracking_uri)
        return jsonify({'success': True, 'type': 'llm', 'message': response_text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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


def get_experiment_total_mase(tracking_uri: str, experiment_name: str) -> Optional[float]:
    """Helper to calculate total weighted MASE for an entire experiment."""
    try:
        results = get_eval_metrics(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            metric_name='mase',
            verbose=False
        )
        if not results:
            return None

        run_ids = [r["eval_run_id"] for r in results if not math.isnan(r["metric_value"])]
        counts = get_timeseries_counts(tracking_uri, run_ids)
        stats = compute_weighted_metric_stats(results, counts)
        return stats["weighted_average"]
    except Exception as e:
        print(f"Error calculating total MASE for {experiment_name}: {e}")
        return None

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """
    1. Global: Best Parent Run from specific core GEF experiments.
    2. Architecture-Specific: Total weighted MASE for all experiments matching ARCH_*.
    """
    try:
        tracking_uri = request.args.get('tracking_uri', DEFAULT_TRACKING_URI)
        cached_payload = _leaderboard_cache.get("payload")
        cached_timestamp = _leaderboard_cache.get("timestamp")
        cached_uri = _leaderboard_cache.get("tracking_uri")
        if (
            cached_payload is not None
            and cached_timestamp is not None
            and cached_uri == tracking_uri
            and (pd.Timestamp.utcnow().timestamp() - cached_timestamp) < LEADERBOARD_CACHE_TTL_SECONDS
        ):
            return jsonify(cached_payload)

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # --- 1. Global Leaderboard (Best Parent Run Logic) ---
        target_global_experiments = ['GEF_MLP', 'GEF_NBEATS', 'GEF_LSTM', 'GEF_LightGBM', 'GEF_TCN']
        global_leaderboard = []
        
        for exp_name in target_global_experiments:
            try:
                mase_results = get_eval_metrics(tracking_uri, exp_name, 'mase', verbose=False)
                if not mase_results: continue

                mase_run_ids = [r["eval_run_id"] for r in mase_results if not math.isnan(r["metric_value"])]
                mase_counts = get_timeseries_counts(tracking_uri, mase_run_ids)

                parent_run_stats = {}
                for r in mase_results:
                    val = r["metric_value"]
                    if math.isnan(val): continue
                    p_name = r["parent_run_name"]
                    p_id = r["parent_run_id"]
                    if p_name not in parent_run_stats:
                        parent_run_stats[p_name] = {'weighted_sum': 0.0, 'total_ts_count': 0}
                    ts_count = mase_counts.get(r["eval_run_id"], 0)
                    if ts_count > 0:
                        parent_run_stats[p_name]['weighted_sum'] += val * ts_count
                        parent_run_stats[p_name]['total_ts_count'] += ts_count

                best_parent_run = None
                best_mase = float('inf')
                for p_name, stats in parent_run_stats.items():
                    if stats['total_ts_count'] > 0:
                        avg_mase = stats['weighted_sum'] / stats['total_ts_count']
                        if avg_mase < best_mase:
                            best_mase = avg_mase
                            best_parent_run = p_name
                
                if best_parent_run:
                    row = {'experiment': exp_name, 'parent_run': best_parent_run, 'mase': best_mase}
                    # Get other metrics for the winner
                    for m in ['mae', 'mape', 'rmse']:
                         other_results = get_eval_metrics(tracking_uri, exp_name, m, parent_filter=best_parent_run, verbose=False)
                         other_run_ids = [r["eval_run_id"] for r in other_results if not math.isnan(r["metric_value"])]
                         other_counts = get_timeseries_counts(tracking_uri, other_run_ids)
                         stats = compute_weighted_metric_stats(other_results, other_counts)
                         row[m] = stats["weighted_average"]
                    global_leaderboard.append(row)
            except Exception as e:
                print(f"Error processing global leaderboard for {exp_name}: {e}")

        global_leaderboard.sort(key=lambda x: x['mase'] if x.get('mase') is not None else float('inf'))

        # --- 2. Architecture Leaderboards (Experiment Total Performance) ---
        arch_keywords = ['MLP', 'NBEATS', 'LSTM', 'LightGBM']
        all_experiments = client.search_experiments()
        architectures_data = {k: [] for k in arch_keywords}
        
        for exp in all_experiments:
            if exp.lifecycle_stage != 'active': continue
            exp_name_upper = exp.name.upper()
            
            for keyword in arch_keywords:
                if keyword in exp_name_upper:
                    total_mase = get_experiment_total_mase(tracking_uri, exp.name)
                    if total_mase is not None:
                        architectures_data[keyword].append({
                            'experiment': exp.name,
                            'mase': total_mase
                        })
                    break # Assign to the first matching architecture
        
        # Sort each architecture group by MASE
        for keyword in arch_keywords:
            architectures_data[keyword].sort(key=lambda x: x['mase'])

        payload = {
            'success': True,
            'global': global_leaderboard,
            'architectures': architectures_data
        }
        _leaderboard_cache["timestamp"] = pd.Timestamp.utcnow().timestamp()
        _leaderboard_cache["payload"] = payload
        _leaderboard_cache["tracking_uri"] = tracking_uri
        return jsonify(payload)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



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
