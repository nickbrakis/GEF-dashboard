import os
import shutil
import tempfile
import pandas as pd
from typing import List, Dict, Optional, Union
import mlflow
from mlflow.tracking import MlflowClient
import argparse
import csv
import math
import requests

def get_eval_metrics(
    tracking_uri: str,
    experiment_name: str,
    metric_name: str = "mase",
    parent_filter: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Retrieve metrics from eval subruns of an MLflow experiment.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
        metric_name: Name of the metric to retrieve (default: 'mase')
        parent_filter: Optional filter string for parent runs
        verbose: If True, print detailed information about each run
    
    Returns:
        List of dictionaries containing run information and metrics
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Get the experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id
    
    if verbose:
        print(f"Found experiment: {experiment_name} (ID: {experiment_id})")
    
    # Get all runs for the experiment
    # We fetch all (or filtered by name if possible, but let's do Python filtering to be safe/exact match as requested)
    # Note: Previously parent_filter was treated as a SQL filter string. Now it is a strict run name.
    
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
        
        # Get child runs of the parent run
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=1000,
        )
        
        for child_run in child_runs:
            child_run_name = child_run.data.tags.get("mlflow.runName", "")
            
            # Check if the run is an eval run and contains the metric
            if "eval" in child_run_name.lower():
                metrics = child_run.data.metrics
                if metric_name in metrics:
                    result = {
                        "parent_run_id": parent_run.info.run_id,
                        "parent_run_name": parent_run_name,
                        "eval_run_id": child_run.info.run_id,
                        "eval_run_name": child_run_name,
                        "metric_name": metric_name,
                        "metric_value": metrics[metric_name]
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
    try:
        # Construct API URL
        # We need to handle potential trailing slashes in tracking_uri
        base_uri = tracking_uri.rstrip('/')
        url = f"{base_uri}/api/2.0/mlflow/artifacts/list"
        
        params = {
            "run_id": run_id,
            "path": "eval_results"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            files = data.get("files", [])
            # Count only directories
            # MLflow API returns 'is_dir' boolean
            directories = [f for f in files if f.get("is_dir")]
            return len(directories)
        else:
            # If path doesn't exist or other error, return 0 or handle accordingly
            return 0
            
    except Exception as e:
        print(f"  ⚠️  Error counting timeseries for run {run_id}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve metrics from eval subruns of MLflow experiments"
    )
    parser.add_argument(
        "--tracking-uri",
        default="https://mlflow.gpu.epu.ntua.gr",
        help="MLflow tracking server URI (default: https://mlflow.gpu.epu.ntua.gr)"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Name of the MLflow experiment"
    )
    parser.add_argument(
        "--metric",
        default="mase",
        help="Name of the metric to retrieve (default: mase)"
    )
    parser.add_argument(
        "--filter",
        help="Filter string for parent runs (e.g., \"tags.mlflow.runName LIKE 'mlp_county_%%'\")"
    )
    parser.add_argument(
        "--csv",
        help="Save results to CSV file with the specified filename"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information about each run"
    )
    # Removing --use-artifacts as we now do this by default/weighted average method
    
    args = parser.parse_args()
    
    try:
        results = get_eval_metrics(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            metric_name=args.metric,
            parent_filter=args.filter,
            verbose=args.verbose
        )
        
        if results:
            # Calculate Weighted Average
            print("\nCalculating weighted average (fetching timeseries counts)...")
            
            valid_results = []
            total_ts_count = 0
            weighted_sum = 0
            
            metric_values = [] # For min/max calculation
            
            for r in results:
                val = r["metric_value"]
                if math.isnan(val):
                    continue
                
                # Fetch count
                ts_count = get_timeseries_count(args.tracking_uri, r["eval_run_id"])
                
                if ts_count > 0:
                    r["ts_count"] = ts_count
                    valid_results.append(r)
                    
                    metric_values.append(val)
                    weighted_sum += val * ts_count
                    total_ts_count += ts_count
                    
                    if args.verbose:
                        print(f"  Run: {r['eval_run_name']} | Metric: {val:.6f} | Count: {ts_count}")
                else:
                    if args.verbose:
                        print(f"  Run: {r['eval_run_name']} | Metric: {val:.6f} | Count: 0 (Ignored)")
            
            print(f"\n{'='*60}")
            print(f"Experiment: {args.experiment}")
            print(f"Metric: {args.metric}")
            print(f"Number of valid eval runs: {len(valid_results)}")
            
            if len(results) > len(valid_results):
                print(f"⚠️  Runs excluded (NaN or 0 count): {len(results) - len(valid_results)}")
            
            if valid_results and total_ts_count > 0:
                weighted_average = weighted_sum / total_ts_count
                
                # Standard (unweighted) average for comparison
                unweighted_average = sum(metric_values) / len(metric_values)
                min_val = min(metric_values)
                max_val = max(metric_values)
                
                print(f"\nWeighted Average: {weighted_average:.6f}")
                print(f"(Unweighted Average: {unweighted_average:.6f})")
                print(f"Min: {min_val:.6f}")
                print(f"Max: {max_val:.6f}")
                print(f"Total Timeseries: {total_ts_count}")
            else:
                print("⚠️  No valid data for calculation.")
            
            print(f"{'='*60}\n")
            
            if args.csv:
                save_to_csv(valid_results, args.csv)
        else:
            print(f"No eval subruns found with metric '{args.metric}'.")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
