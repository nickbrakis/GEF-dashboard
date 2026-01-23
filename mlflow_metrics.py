import mlflow
from mlflow.tracking import MlflowClient
import argparse
import csv
import math
from typing import List, Dict, Optional

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
    filter_string = parent_filter if parent_filter else ""
    parent_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1000,
    )
    
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

def save_to_csv(results: List[Dict], filename: str):
    """Save results to a CSV file."""
    if not results:
        print("No results to save.")
        return
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {filename}")

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
            metric_values = [r["metric_value"] for r in results]
            
            # Filter out NaN values
            valid_values = [v for v in metric_values if not math.isnan(v)]
            nan_count = len(metric_values) - len(valid_values)
            
            print(f"\n{'='*60}")
            print(f"Experiment: {args.experiment}")
            print(f"Metric: {args.metric}")
            print(f"Number of eval runs: {len(results)}")
            
            if nan_count > 0:
                print(f"⚠️  NaN values found: {nan_count}")
                print(f"Valid values: {len(valid_values)}")
            
            if valid_values:
                average = sum(valid_values) / len(valid_values)
                min_val = min(valid_values)
                max_val = max(valid_values)
                
                print(f"Average: {average:.6f}")
                print(f"Min: {min_val:.6f}")
                print(f"Max: {max_val:.6f}")
            else:
                print("⚠️  No valid metric values found (all values are NaN)")
            
            print(f"{'='*60}\n")
            
            if args.csv:
                save_to_csv(results, args.csv)
        else:
            print(f"No eval subruns found with metric '{args.metric}'.")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
