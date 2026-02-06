import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

app = Flask(__name__)

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.gpu.epu.ntua.gr")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
LOG_MCP = os.environ.get("MCP_LOG", "0") == "1"


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


def tool_ping(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok"}


TOOLS = {
    "mlflow.list_experiments": tool_list_experiments,
    "mlflow.list_runs": tool_list_runs,
    "mlflow.get_run": tool_get_run,
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
