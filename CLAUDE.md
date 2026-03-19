# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Target
- Dashboard that aims to be: user-friendly, interactive, convenient, providing graphs and insights to the scientist. A better connection with Mlflow Tracking Server

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Development (port 5000)
python app.py

# Production
gunicorn --bind 0.0.0.0:5000 app:app

# MCP server — must run separately on port 7001
python mcp_server.py
```

Requires a `.env` file with:
- `MLFLOW_TRACKING_URI` — MLflow server URL
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` — MinIO/S3 access
- `OPENAI_API_KEY`, `OPENAI_MODEL` — Chat LLM
- `MCP_BASE_URL`, `MCP_TIMEOUT_SECONDS` — MCP server connection
- `LEADERBOARD_CACHE_TTL_SECONDS`, `CHAT_MEMORY_MAX_TURNS` — tuning knobs

## Architecture

Two Flask processes, one frontend:

**`app.py`** — Main web server (REST API + static file serving)
- Talks to MLflow tracking server and MinIO/S3 for artifact CSVs
- Talks to OpenAI API for chat (with tool calling)
- Proxies tool calls to `mcp_server.py` for LLM integration

**`mcp_server.py`** — MCP JSON-RPC server (port 7001)
- Implements MCP protocol 2024-11-05 over HTTP POST at `/mcp`
- Exposes 5 MLflow tools to the LLM: `list_experiments`, `list_runs`, `get_run`, `list_pipeline_runs`, `get_experiment_evaluation`
- `get_experiment_evaluation` is the core tool: computes weighted/unweighted MASE metrics by fetching eval CSVs from MinIO

**`static/`** — Single-page frontend
- `index.html`: 5 navigation modes (Single Experiment, Compare, Leaderboard, Plot Comparison, Chat)
- `app.js`: All client-side logic and API calls (~1000 lines)
- `style.css`: Glass-morphism dark theme

## Key Data Flow

**Metrics computation**: MLflow run → eval artifacts in MinIO (`eval_results/` CSVs) → aggregated by timeseries count (weighted) or simple average (unweighted). GEF-prefixed experiments use unweighted by default; others use weighted.

**Leaderboard**: Hardcoded core experiments (`GEF_MLP`, `GEF_NBEATS`, `GEF_LSTM`, `GEF_LightGBM`, `GEF_TCN`). Results cached (default 15 min).

**Chat**: User message → OpenAI with function calling → tool calls forwarded to MCP server → MLflow data → response with conversation memory (default 10 turns).

## Mlflow Experiment Structure
- These experiments are a part of a Short Term Load Forecasting study and the main methods that are compared are global, semi global and local methods. In global methods a single model is used to be trained and evaluated with all Timeseries, semi global uses on model to be trained on a group of Timeseries and local methods use on single model for every single Timeseries
- Every run in Mlflow Experiments represent one full ML pipeline run with stages (load, etl, train, eval). Each stage is implemented as child run that contains useful artifacts.
- The experiments on Mlflow don't follow the same pattern.  
- The experiments with names that follow the pattern GEF_<model>, represent the runs of hyperparameter tuning process of the global models. One of them has the actual valueable information that is the best mase value of the eval stage child run
- The other experiments are the local and semi global approaches. All of the runs of those experiments are valuable, they represent the run of the training and evaluation process of each model of some group of Timeseries


## No Test Suite

There are no automated tests. `test_data/inspect_test.py` is a one-off data inspection script, not a test runner.
