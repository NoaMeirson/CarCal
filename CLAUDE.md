# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CarCal is a car damage assessment system using ML to detect car parts and damage from images. It consists of three independently deployable FastAPI microservices:

```
Client (port 8000) → API (port 8002) → Engine (port 8001)
```

- **Client**: Accepts image file uploads, base64-encodes and forwards to API
- **API**: Orchestrates/validates requests between Client and Engine
- **Engine**: Core ML inference — loads and runs two models on startup, returns polygon detections

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# If adding new dependencies:
pip freeze > requirements.txt
```

## Running Services

```bash
# Individual services
uvicorn Client.Client_main:app --port 8000 --reload
uvicorn Engine.Engine_main:app --port 8001 --reload
uvicorn API.API_main:app --port 8002 --reload
```

VS Code: Use the "Run All Services" compound launch config, which also opens the dashboard at `http://localhost:8002/services`.

## Architecture

### Data Flow

1. User POSTs image file → Client `/analyze`
2. Client base64-encodes image, assigns UUID, POSTs `ClientAnalyzeRequest` → API `/analyze`
3. API validates (size/format), POSTs `EngineAnalyzeRequest` → Engine `/process`
4. Engine decodes image, runs ML model(s), converts masks → polygons, returns `EngineAnalyzeResponse`
5. Response propagates back to user

### Shared Models (`models.py`)

All three services import from root-level `models.py`. Key types:
- `ClientAnalyzeRequest` / `ClientAnalyzeResponse` — Client ↔ API contract
- `EngineAnalyzeRequest` / `EngineAnalyzeResponse` — API ↔ Engine contract
- `Detection(id, type, label, confidence, polygon, matches)` — single detected object
- `MatchInfo(part, score, coveragePercent)` — damage-to-part overlap result

### Engine ML Pipeline

The Engine loads two models at startup (cached in module globals throughout service lifetime):

**Car Parts Model** (Mask2Former, HuggingFace Transformers):
- Path: `Engine/artifacts/car_parts/car_parts_M2F_model/` (git-ignored)
- Performs instance segmentation to detect individual car parts

**Damage Model** (YOLO segmentation, Ultralytics):
- Path: `Engine/artifacts/damages/damage_YOLO_model.pt`
- Detects damage types (scratch, dent, crack, etc.) with confidence ≥ 0.25

**Processing Modes** (set in `Client/ClientConfig.py`):
- `car_parts_only` — Mask2Former only
- `damage_only` — YOLO only
- `combined` — Both models; `Engine/services/combine_service.py` overlaps masks to associate each damage with the car part(s) it affects, computing `score` (% of damage on that part) and `coveragePercent` (% of part covered by damage)

Mask-to-polygon conversion uses OpenCV contours in `Engine/utils/segmentation_utils.py`.

### Configuration Files

- `Client/ClientConfig.py` — `API_URL` (currently hardcoded to external IP), `MODE`
- `API/APIConfig.py` — `ENGINE_URL`, `REQUEST_TIMEOUT`, `MAX_IMAGE_SIZE`
- `Engine/EngineConfig.py` — model paths, `DEVICE`, `DAMAGE_MODEL_CONFIDENCE`

### Health Checks

- `GET /health` on each service
- `GET http://localhost:8001/health` returns extended `EngineHealthResponse` with per-model readiness
- Engine starts in degraded state if models fail to load

## Key Notes

- **No requirements.txt in repo by default** — generate with `pip freeze > requirements.txt`
- **ML model artifacts not in repo** — `Engine/artifacts/car_parts/car_parts_M2F_model/` is git-ignored; the YOLO `.pt` file is committed
- **`React_Client/`** — empty placeholder for a future React frontend
- **Async/sync mix** — Client handler is `async`, API and Engine handlers are sync
- Images are transported as base64 strings over HTTP between all services
