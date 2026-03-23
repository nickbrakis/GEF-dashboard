# GEF Dashboard

A web dashboard for tracking and analyzing MLflow experiments from a **Short Term Load Forecasting (STLF)** research study. It provides an interactive UI to visualize metrics, compare forecasting methods, and query experiment results in natural language.

## Features

- **Single Experiment** — load and visualize metrics for any MLflow experiment
- **Compare** — side-by-side metric comparison across multiple experiments
- **Global vs Grouped** — plot comparison between global and semi-global/local models per timeseries
- **Leaderboard** — ranked table of global models (GEF_*) and architecture-specific rankings (MLP, NBEATS, LSTM, TCN)
- **Chat** — natural language interface to query MLflow data, powered by OpenAI with tool calling via an MCP server

## Architecture

Two Flask processes + a single-page frontend:

| Component | File | Port |
|-----------|------|------|
| Main web server | `app.py` | 5000 |
| MCP tool server | `mcp_server.py` | 7001 |
| Frontend | `static/` | — |

The main server handles REST API calls, talks to MLflow and MinIO/S3, and proxies LLM tool calls to the MCP server. The MCP server exposes MLflow as structured tools for the LLM.

## Setup

### Requirements

- Python 3.11+
- Access to an MLflow Tracking Server
- MinIO/S3 storage for eval artifacts
- OpenAI API key

## Study Context

The dashboard is built around a STLF study comparing three forecasting paradigms:

- **Global** (`GEF_*` experiments) — one model trained on all timeseries; only the best hyperparameter run matters
- **Semi-global** (`NBEATS_*`, `MLP_*`, etc.) — one model per cluster of timeseries
- **Local** (`MLP_*`, `LSTM_*`, etc.) — one model per individual timeseries

Each MLflow run represents a full pipeline: `load_data → etl → train → eval`. The `eval` child run holds the MASE metric and evaluation artifacts used for all computations.
