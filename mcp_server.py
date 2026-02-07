import json
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
import mlflow
from mlflow.tracking import MlflowClient
import requests

load_dotenv()

app = Flask(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.gpu.epu.ntua.gr")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
LOG_MCP = os.environ.get("MCP_LOG", "0") == "1"
_timeseries_count_cache: Dict[tuple, int] = {}


def _log(message: str) -> None:
    if not LOG_MCP:
        return
    print(message, flush=True)


def _set_tracking_uri(tracking_uri: Optional[str]) -> str:
    uri = (tracking_uri or "").strip()
    if not uri:
        uri = DEFAULT_TRACKING_URI
    # Guard against accidental local path values (e.g. "mlruns")
    if uri in {"mlruns", "mlruns/"}:
        uri = DEFAULT_TRACKING_URI
    # If the URI doesn't look like a valid scheme, fall back to default
    if "://" not in uri and not uri.startswith("file:"):
        uri = DEFAULT_TRACKING_URI
    # Guard against accidental localhost from the web app
    if uri.startswith("http://localhost:5000") or uri.startswith("http://127.0.0.1:5000"):
        uri = DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    return uri


def _resolve_experiment(
    client: MlflowClient,
    experiment_name: Optional[str],
    experiment_id: Optional[str],
) -> Any:
    if experiment_id:
        exp = client.get_experiment(str(experiment_id))
        if not exp:
            raise ValueError(f"Experiment id '{experiment_id}' not found")
        return exp
    if not experiment_name:
        raise ValueError("experiment_name or experiment_id is required")
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    return exp


def _run_name(run: Any) -> str:
    return run.data.tags.get("mlflow.runName") or ""


def _is_parent_run(run: Any) -> bool:
    return not run.data.tags.get("mlflow.parentRunId")


def _stage_from_run_name(run_name: str) -> str:
    n = (run_name or "").strip().lower()
    if n.startswith("load_data"):
        return "load_data"
    if n.startswith("etl"):
        return "etl"
    if n.startswith("train_"):
        return "train"
    if "eval" in n:
        return "eval"
    return "other"


def _get_timeseries_count(tracking_uri: str, run_id: str) -> int:
    cache_key = (tracking_uri, run_id)
    cached = _timeseries_count_cache.get(cache_key)
    if cached is not None:
        return cached

    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/api/2.0/mlflow/artifacts/list"
    params = {"run_id": run_id, "path": "eval_results"}
    timeouts = [(5, 20), (5, 30), (10, 45)]

    for timeout in timeouts:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                files = response.json().get("files", [])
                count = len([f for f in files if f.get("is_dir")])
                _timeseries_count_cache[cache_key] = count
                return count
            if response.status_code in {403, 404}:
                _timeseries_count_cache[cache_key] = 0
                return 0
        except Exception:
            continue

    _timeseries_count_cache[cache_key] = 0
    return 0


def _get_timeseries_counts(tracking_uri: str, run_ids: List[str], max_workers: int = 12) -> Dict[str, int]:
    unique_ids = list({r for r in run_ids if r})
    if not unique_ids:
        return {}

    counts: Dict[str, int] = {}
    worker_count = min(max_workers, len(unique_ids))
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {ex.submit(_get_timeseries_count, tracking_uri, r): r for r in unique_ids}
        for future in as_completed(futures):
            rid = futures[future]
            try:
                counts[rid] = future.result()
            except Exception:
                counts[rid] = 0
    return counts


def _summarize_metric_values(
    values: List[float],
    eval_run_ids: List[str],
    counts: Dict[str, int],
    use_weighted: bool,
) -> Dict[str, Any]:
    valid: List[float] = []
    weighted_sum = 0.0
    total_ts = 0

    for idx, val in enumerate(values):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        valid.append(float(val))
        if use_weighted:
            rid = eval_run_ids[idx]
            ts_count = counts.get(rid, 0)
            if ts_count > 0:
                weighted_sum += float(val) * ts_count
                total_ts += ts_count

    summary: Dict[str, Any] = {
        "valid_count": len(valid),
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
        "unweighted_average": (sum(valid) / len(valid)) if valid else None,
        "weighted_average": None,
        "total_timeseries": total_ts if use_weighted else None,
    }

    if use_weighted and total_ts > 0:
        summary["weighted_average"] = weighted_sum / total_ts
    return summary


def tool_list_experiments(params: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = _set_tracking_uri(params.get("tracking_uri"))
    _log(f"[mcp] list_experiments tracking_uri={tracking_uri}")
    client = MlflowClient()
    experiments = client.search_experiments()
    items = [
        {
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "lifecycle_stage": exp.lifecycle_stage,
        }
        for exp in experiments
        if exp.lifecycle_stage == "active"
    ]
    return {
        "tracking_uri": tracking_uri,
        "experiments": items,
    }


def tool_list_runs(params: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = _set_tracking_uri(params.get("tracking_uri"))
    _log(f"[mcp] list_runs tracking_uri={tracking_uri} params={params}")
    client = MlflowClient()

    experiment_name = params.get("experiment_name")
    experiment_id = params.get("experiment_id")
    max_results = int(params.get("max_results", 50))
    filter_string = params.get("filter_string", "")
    order_by = params.get("order_by")

    if not experiment_id:
        if not experiment_name:
            raise ValueError("experiment_name or experiment_id is required")
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=order_by,
    )

    items = []
    for run in runs:
        items.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName"),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        )

    return {
        "tracking_uri": tracking_uri,
        "experiment_id": experiment_id,
        "runs": items,
    }


def tool_get_run(params: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = _set_tracking_uri(params.get("tracking_uri"))
    _log(f"[mcp] get_run tracking_uri={tracking_uri} params={params}")
    client = MlflowClient()

    run_id = params.get("run_id")
    if not run_id:
        raise ValueError("run_id is required")

    run = client.get_run(run_id)
    return {
        "tracking_uri": tracking_uri,
        "run": {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        },
    }


def tool_list_pipeline_runs(params: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = _set_tracking_uri(params.get("tracking_uri"))
    _log(f"[mcp] list_pipeline_runs tracking_uri={tracking_uri} params={params}")
    client = MlflowClient()

    experiment = _resolve_experiment(client, params.get("experiment_name"), params.get("experiment_id"))
    max_results = int(params.get("max_results", 50))
    include_children = bool(params.get("include_children", True))
    parent_name_contains = (params.get("parent_name_contains") or "").strip().lower()

    # Pull a larger run set, then keep parent runs only.
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=max(max_results * 10, 500),
        order_by=["attributes.start_time DESC"],
    )
    parent_runs = [r for r in all_runs if _is_parent_run(r)]
    if parent_name_contains:
        parent_runs = [r for r in parent_runs if parent_name_contains in _run_name(r).lower()]
    parent_runs = sorted(parent_runs, key=lambda r: r.info.start_time or 0, reverse=True)[:max_results]

    parent_items: List[Dict[str, Any]] = []
    for parent in parent_runs:
        item: Dict[str, Any] = {
            "run_id": parent.info.run_id,
            "run_name": _run_name(parent),
            "status": parent.info.status,
            "start_time": parent.info.start_time,
            "end_time": parent.info.end_time,
        }
        if include_children:
            children = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                max_results=1000,
            )
            child_items = []
            stages = {"load_data": 0, "etl": 0, "train": 0, "eval": 0, "other": 0}
            for child in children:
                child_name = _run_name(child)
                stage = _stage_from_run_name(child_name)
                stages[stage] = stages.get(stage, 0) + 1
                child_items.append(
                    {
                        "run_id": child.info.run_id,
                        "run_name": child_name,
                        "stage": stage,
                        "status": child.info.status,
                        "start_time": child.info.start_time,
                        "end_time": child.info.end_time,
                        "metrics": child.data.metrics,
                    }
                )
            item["children"] = child_items
            item["stage_counts"] = stages
            item["has_expected_4_stages"] = all(stages.get(k, 0) >= 1 for k in ["load_data", "etl", "train", "eval"])
        parent_items.append(item)

    return {
        "tracking_uri": tracking_uri,
        "experiment": {
            "name": experiment.name,
            "experiment_id": experiment.experiment_id,
        },
        "parent_runs": parent_items,
    }


def tool_get_experiment_evaluation(params: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = _set_tracking_uri(params.get("tracking_uri"))
    _log(f"[mcp] get_experiment_evaluation tracking_uri={tracking_uri} params={params}")
    client = MlflowClient()

    experiment = _resolve_experiment(client, params.get("experiment_name"), params.get("experiment_id"))
    metrics = params.get("metrics") or ["mase"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty array")
    aggregate_mode = (params.get("aggregate_mode") or "auto").lower()
    if aggregate_mode not in {"auto", "weighted", "unweighted"}:
        raise ValueError("aggregate_mode must be one of: auto, weighted, unweighted")
    parent_run_name = params.get("parent_run_name")
    include_parent_breakdown = bool(params.get("include_parent_breakdown", False))

    parent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=3000,
    )
    parent_runs = [r for r in parent_runs if _is_parent_run(r)]
    if parent_run_name:
        parent_runs = [r for r in parent_runs if _run_name(r) == parent_run_name]

    eval_rows: List[Dict[str, Any]] = []
    for parent in parent_runs:
        children = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=1000,
        )
        for child in children:
            child_name = _run_name(child)
            if _stage_from_run_name(child_name) != "eval":
                continue
            eval_rows.append(
                {
                    "parent_run_id": parent.info.run_id,
                    "parent_run_name": _run_name(parent),
                    "eval_run_id": child.info.run_id,
                    "eval_run_name": child_name,
                    "metrics": child.data.metrics,
                }
            )

    use_weighted = False
    if aggregate_mode == "weighted":
        use_weighted = True
    elif aggregate_mode == "auto":
        use_weighted = not experiment.name.upper().startswith("GEF")

    eval_run_ids = [row["eval_run_id"] for row in eval_rows]
    counts = _get_timeseries_counts(tracking_uri, eval_run_ids) if use_weighted else {}

    metric_summary: Dict[str, Any] = {}
    for metric in metrics:
        values = [row["metrics"].get(metric) for row in eval_rows]
        summary = _summarize_metric_values(values, eval_run_ids, counts, use_weighted)

        used = "weighted" if use_weighted else "unweighted"
        fallback_reason = None
        if use_weighted and summary["weighted_average"] is None and summary["unweighted_average"] is not None:
            used = "unweighted"
            fallback_reason = "No timeseries counts available from eval artifacts."

        metric_summary[metric] = {
            "aggregation_used": used,
            "fallback_reason": fallback_reason,
            "total_eval_runs": len(eval_rows),
            "valid_eval_runs": summary["valid_count"],
            "weighted_average": summary["weighted_average"],
            "unweighted_average": summary["unweighted_average"],
            "average": summary["weighted_average"] if used == "weighted" else summary["unweighted_average"],
            "min": summary["min"],
            "max": summary["max"],
            "total_timeseries": summary["total_timeseries"] if use_weighted else None,
        }

    parent_breakdown: List[Dict[str, Any]] = []
    if include_parent_breakdown:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in eval_rows:
            grouped.setdefault(row["parent_run_name"], []).append(row)
        for p_name, rows in grouped.items():
            row_eval_ids = [r["eval_run_id"] for r in rows]
            row_counts = _get_timeseries_counts(tracking_uri, row_eval_ids) if use_weighted else {}
            per_metric: Dict[str, Any] = {}
            for metric in metrics:
                values = [r["metrics"].get(metric) for r in rows]
                summary = _summarize_metric_values(values, row_eval_ids, row_counts, use_weighted)
                used = "weighted" if use_weighted and summary["weighted_average"] is not None else "unweighted"
                per_metric[metric] = {
                    "average": summary["weighted_average"] if used == "weighted" else summary["unweighted_average"],
                    "weighted_average": summary["weighted_average"],
                    "unweighted_average": summary["unweighted_average"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "valid_eval_runs": summary["valid_count"],
                    "total_timeseries": summary["total_timeseries"] if use_weighted else None,
                    "aggregation_used": used,
                }
            parent_breakdown.append(
                {
                    "parent_run_name": p_name,
                    "total_eval_runs": len(rows),
                    "metrics": per_metric,
                }
            )

    return {
        "tracking_uri": tracking_uri,
        "experiment": {
            "name": experiment.name,
            "experiment_id": experiment.experiment_id,
        },
        "requested_metrics": metrics,
        "aggregate_mode": aggregate_mode,
        "metric_summary": metric_summary,
        "parent_breakdown": parent_breakdown,
    }


def tool_ping(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok"}


TOOLS = {
    "mlflow.list_experiments": tool_list_experiments,
    "mlflow.list_runs": tool_list_runs,
    "mlflow.get_run": tool_get_run,
    "mlflow.list_pipeline_runs": tool_list_pipeline_runs,
    "mlflow.get_experiment_evaluation": tool_get_experiment_evaluation,
    "mlflow.ping": tool_ping,
}


def _tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "name": "mlflow.list_experiments",
            "description": "List active MLflow experiments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string", "description": "MLflow tracking URI"},
                },
            },
        },
        {
            "name": "mlflow.list_runs",
            "description": "List runs for an experiment by name or id.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string", "description": "MLflow tracking URI"},
                    "experiment_name": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "max_results": {"type": "integer", "default": 50},
                    "filter_string": {"type": "string"},
                    "order_by": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "mlflow.get_run",
            "description": "Get full details for a run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string", "description": "MLflow tracking URI"},
                    "run_id": {"type": "string"},
                },
                "required": ["run_id"],
            },
        },
        {
            "name": "mlflow.list_pipeline_runs",
            "description": "List parent pipeline runs and optionally their child stages (load_data, etl, train_<model>, eval).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string", "description": "MLflow tracking URI"},
                    "experiment_name": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "max_results": {"type": "integer", "default": 50},
                    "include_children": {"type": "boolean", "default": True},
                    "parent_name_contains": {"type": "string"},
                },
            },
        },
        {
            "name": "mlflow.get_experiment_evaluation",
            "description": "Compute experiment-level eval metrics using unweighted/weighted aggregation. In auto mode, experiments with prefix 'GEF' use unweighted and others use weighted by timeseries count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tracking_uri": {"type": "string", "description": "MLflow tracking URI"},
                    "experiment_name": {"type": "string"},
                    "experiment_id": {"type": "string"},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "aggregate_mode": {"type": "string", "enum": ["auto", "weighted", "unweighted"], "default": "auto"},
                    "parent_run_name": {"type": "string"},
                    "include_parent_breakdown": {"type": "boolean", "default": False},
                },
            },
        },
        {
            "name": "mlflow.ping",
            "description": "Lightweight connectivity test.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def _error_response(req_id: Any, code: int, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _result_response(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


@app.post("/mcp")
def mcp_endpoint():
    req = request.get_json(force=True, silent=True) or {}
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "initialize":
        return jsonify(
            _result_response(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mlflow-mcp", "version": "0.1.0"},
                },
            )
        )
    if method == "tools/list":
        return jsonify(_result_response(req_id, {"tools": _tools_schema()}))
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = TOOLS.get(name)
        if not tool:
            return jsonify(_error_response(req_id, -32601, f"Unknown tool: {name}"))
        try:
            result = tool(args)
            return jsonify(
                _result_response(
                    req_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=True),
                            }
                        ]
                    },
                )
            )
        except Exception as exc:
            return jsonify(_error_response(req_id, -32000, "Tool execution error", {"detail": str(exc)}))
    if method == "ping":
        return jsonify(_result_response(req_id, {"status": "ok"}))

    return jsonify(_error_response(req_id, -32601, f"Method not found: {method}"))


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/debug")
def debug():
    return jsonify(
        {
            "tracking_uri": mlflow.get_tracking_uri(),
            "default_tracking_uri": DEFAULT_TRACKING_URI,
            "env_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
            "mlflow_version": getattr(mlflow, "__version__", "unknown"),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "7001"))
    app.run(host="0.0.0.0", port=port)
