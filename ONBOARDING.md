# CarCal — Onboarding

CarCal detects car parts and damage in a photo. Today the system is two Python
FastAPI microservices (**API**, **Engine**) plus a **React** desktop/web
frontend (`React_Client/`) that talks to them.

```
React_Client (browser or packaged .exe)  →  API (port 8002)  →  Engine (port 8001)
```

> **Note on `Client/`**: CLAUDE.md and the `.vscode` launch config still
> describe a third Python service, `Client/` (port 8000, multipart file
> upload → forwards to API). It has been removed from the working tree —
> `React_Client` now plays that role directly, posting JSON straight to the
> API service. Treat any mention of a Python "Client" service elsewhere in
> the repo's docs as stale.

---

## 1. Shared contract — `models.py`

All three parts of the system import from one root-level file, so request/response
shapes are identical on both hops (API↔Engine payloads are structurally the
same as the old Client↔API ones):

```python
class Point(BaseModel):
    x: float
    y: float

class Polygon(BaseModel):
    points: list[Point]

class ImageInfo(BaseModel):
    width: int
    height: int

class MatchInfo(BaseModel):
    part: str
    score: float                      # 0-1, fraction of the damage on this part
    coveragePercent: float | None      # 0-100, % of the part covered by damage

class Detection(BaseModel):
    id: str
    type: str                          # "part" | "damage"
    label: str
    confidence: float
    polygon: Polygon                   # pixel coords in the original image
    matches: list[MatchInfo] | None    # only populated in "combined" mode

class ClientAnalyzeRequest(BaseModel):   # what React_Client sends to API
    requestId: str
    FileName: str | None = None
    imageBase64: str
    mode: str                            # "car_parts_only" | "damage_only" | "combined"

class EngineAnalyzeRequest(BaseModel):   # what API sends to Engine (same shape)
    requestId: str
    FileName: str | None = None
    imageBase64: str
    mode: str

class ClientAnalyzeResponse(BaseModel):  # API's reply to React_Client
    requestId: str
    FileName: str | None = None
    status: str                          # "ok" | "error"
    mode: str
    image: ImageInfo | None = None
    detections: list[Detection]
    message: str | None = None

class EngineAnalyzeResponse(BaseModel):  # Engine's reply to API (same shape)
    ...  # identical fields to ClientAnalyzeResponse

class HealthResponse(BaseModel):
    status: str

class EngineHealthResponse(BaseModel):
    status: str
    engineReady: bool
    carPartsModelReady: bool
    damageModelReady: bool
    message: str | None = None
```

`polygon.points` are raw pixel coordinates in the *original uploaded image's*
coordinate space — no normalization to 0–1.

---

## 2. API service — `API/`

The orchestrator. Validates the incoming request, forwards it to Engine, and
relays the result back. Runs on **port 8002**.

### `API/API_main.py`
```python
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["Content-Type"])

POST /analyze   → analyze_endpoint(request: ClientAnalyzeRequest) -> ClientAnalyzeResponse
GET  /health    → HealthResponse(status="ok")
GET  /services  → services_dashboard()   # HTML dashboard, see §5
```
This is the **only service with CORS enabled** — it needs to be, since
`React_Client` runs in a browser on a different origin/port and calls this
service's `/analyze` directly.

### `API/APIConfig.py`
```python
ENGINE_URL = "http://localhost:8001/process"
ENGINE_HEALTH_URL = "http://localhost:8001/health"   # declared, unused elsewhere
REQUEST_TIMEOUT = 60
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10MB
IMAGE_SIZE = 640                     # declared, unused elsewhere
```

### `API/API_methods.py`
```python
def analyze(request: ClientAnalyzeRequest) -> ClientAnalyzeResponse:
    engine_result = send_to_engine(request)
    return ClientAnalyzeResponse(requestId=..., status=engine_result["status"], ...)

def validate_image(image_bytes: bytes) -> None:
    # raises ValueError if empty, over MAX_IMAGE_SIZE, or not a real image
    # (PIL Image.open(...).verify())

def send_to_engine(request) -> dict:
    # 1. base64-decode request.imageBase64 (ValueError if invalid)
    # 2. validate_image(image_bytes)
    # 3. re-encode to base64, build EngineAnalyzeRequest
    # 4. POST to ENGINE_URL, timeout=REQUEST_TIMEOUT
    # 5. raise RuntimeError("Engine service error") if not 200
    # 6. return response.json()
```
No custom exception handlers are registered anywhere — any `ValueError`/
`RuntimeError` raised here becomes FastAPI's default plain 500, not a
structured error body.

> **Known bug**: `send_to_engine` builds the outgoing request as
> `EngineAnalyzeRequest(requestId=..., filename=request.FileName, ...)` —
> note the lowercase `filename` keyword, but the Pydantic field is
> `FileName`. Pydantic v2 silently ignores unknown kwargs by default, so this
> doesn't error — it just drops the filename. **The original filename never
> reaches the Engine**, and `FileName` is always `None` from that point on.

### `API/dashboard.py`
Two independent things:
- `check_service_health(url) -> str` — 1s-timeout GET, returns `"UP"` /
  `"TIMEOUT"` / `"CONNECTION_FAILED"` / `"UNEXPECTED_STATUS_{code}"` /
  `"ERROR"`.
- `services_dashboard()` — returns a raw `HTMLResponse` with three cards
  (Client, API, Engine), each pinging `http://localhost:{8000,8002,8001}/health`
  and linking to that service's `/docs` and `/health`. API's own card is
  hardcoded `"UP"` (it can't meaningfully self-report down). **The Client
  card still checks port 8000**, which nothing serves anymore now that
  `Client/` is gone — expect that card to always read `CONNECTION_FAILED`.
- `run_pipeline_test()` — a synthetic end-to-end health check (pings Client
  and Engine `/health`, then POSTs a fake payload to API `/analyze`). **Not
  wired to any route** — dead code, callable only from a Python shell.

---

## 3. Engine service — `Engine/`

Runs the actual ML inference. Loads two models once at startup and keeps
them cached in module-level globals for the process lifetime. Runs on
**port 8001**.

### `Engine/EngineConfig.py`
```python
DEVICE = "cpu"                 # declared, NOT actually used (see below)
MAX_DETECTIONS = 20            # declared, NOT enforced anywhere

CAR_PARTS_MODEL_DIR = Engine/artifacts/car_parts/car_parts_M2F_model/
CAR_PARTS_MODEL_PREFERRED_DEVICE = "cuda"

DAMAGE_MODEL_PATH = Engine/artifacts/damages/damage_YOLO_model.pt
DAMAGE_MODEL_PREFERRED_DEVICE = "cuda"
DAMAGE_MODEL_CONFIDENCE = 0.25  # declared, NOT actually passed to YOLO (see below)
```
Each model independently picks `"cuda"` if its `*_PREFERRED_DEVICE == "cuda"`
**and** `torch.cuda.is_available()`, else falls back to `"cpu"`. The
top-level `DEVICE` constant is never consulted.

> **Known bug**: `DAMAGE_MODEL_CONFIDENCE` is defined but never passed as
> `conf=` to `YOLO.predict()` — the damage model runs at Ultralytics'
> default confidence threshold, not 0.25. (CLAUDE.md's claim of "confidence
> ≥ 0.25" is currently inaccurate.)

### `Engine/Engine_main.py`
```python
@app.on_event("startup")
def startup_event():
    try: load_car_parts_model()
    except Exception as exc: print(f"[Engine startup] Failed to load car parts model: {exc}")
    try: load_damage_model()
    except Exception as exc: print(f"[Engine startup] Failed to load damage model: {exc}")

POST /process → process_endpoint(request: EngineAnalyzeRequest) -> EngineAnalyzeResponse
GET  /health  → engine_health() -> EngineHealthResponse
```
Each model load is wrapped in its own try/except, so if one model fails to
load the other still comes up and the service still starts — this is the
"degraded" state CLAUDE.md refers to. `engineReady` in the health response is
always `True` (the process is up); only `carPartsModelReady`/
`damageModelReady` reflect real model state, with an error message
concatenating both models' load errors when either is not ready.

### `Engine/Engine_methods.py` — the `/process` pipeline
```python
def process(request: EngineAnalyzeRequest) -> EngineAnalyzeResponse:
    try:
        image = decode_base64_image(request.imageBase64)
        if request.mode == "car_parts_only": return process_parts_only(request, image)
        if request.mode == "damage_only":    return process_damage_only(request, image)
        if request.mode == "combined":       return process_combined(request, image)
        raise ValueError(f"Unsupported mode: {request.mode}")
    except Exception as exc:
        return EngineAnalyzeResponse(
            requestId=request.requestId, FileName=request.FileName,
            status="error", image=None, detections=[], message=str(exc)
        )   # ⚠️ see bug below
```
- **`process_parts_only`**: `run_car_parts_model` → `postprocess_car_parts_raw_outputs`
  → `convert_car_parts_result_to_detections`.
- **`process_damage_only`**: `run_damage_model` → `postprocess_damage_raw_outputs`
  → `build_damage_detections`.
- **`process_combined`**: runs **both models sequentially** (not parallel),
  then `combine_results(damage_result=..., car_parts_result=...)`.

> **Known bug**: the `except` branch's `EngineAnalyzeResponse(...)` call
> omits `mode=`, but `EngineAnalyzeResponse.mode` has no default in
> `models.py` — constructing it without `mode` raises a Pydantic
> `ValidationError` **inside the except block itself**. So the intended
> graceful `status: "error"` response can never actually be returned; any
> exception during processing (bad base64, unsupported mode, model not
> loaded, etc.) surfaces to the client as a raw 500, not the friendly error
> payload the code appears to be trying to produce.

### `Engine/utils/image_utils.py`
```python
def decode_base64_image(image_base64: str) -> Image.Image:
    # base64-decode, then decode_image_bytes; wraps any failure as ValueError

def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    # PIL.Image.open(...).convert("RGB")  — always forces RGB (drops alpha/CMYK)
```

### `Engine/utils/segmentation_utils.py`
```python
def mask_to_polygon(mask: np.ndarray) -> Polygon | None:
    # cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    # keeps only the largest-area contour; returns None if <3 points
```
`RETR_EXTERNAL` means only the outer boundary is kept (holes ignored), and
only the single largest contour becomes the polygon — if a mask has multiple
disconnected blobs, the smaller ones are silently dropped. Shared by both
model services and by `combine_service.py`.

### `Engine/services/car_parts_model_service.py` — Mask2Former (HuggingFace)
Module globals: `_MODEL`, `_PROCESSOR`, `_DEVICE`, `_MODEL_READY`, `_MODEL_LOAD_ERROR`.

- `load_car_parts_model()` — idempotent; lazy-imports `transformers`; raises
  `FileNotFoundError` if `CAR_PARTS_MODEL_DIR` is missing; loads
  `AutoImageProcessor` + `Mask2FormerForUniversalSegmentation` from that dir,
  `.to(device).eval()`. On any failure, resets all globals and re-raises.
- `is_car_parts_model_ready() -> bool`
- `get_car_parts_model_status() -> {ready, device, modelDir, error}`
- `run_car_parts_model(image)` — processor → tensors to device →
  `model(**inputs)` under `torch.no_grad()` → raw HF output.
- `postprocess_car_parts_raw_outputs(raw, image)` — calls
  `processor.post_process_instance_segmentation(raw, target_sizes=[(h, w)])`,
  resizing the segmentation map back to the original image size.
- `get_car_parts_id2label() -> {int: str}` — from `model.config.id2label`.
- `convert_car_parts_result_to_detections(result) -> list[Detection]` — for
  each `segments_info` entry, builds a boolean mask (`segmentation_map ==
  segment_id`), converts to polygon, skips if `None`; `type="part"`,
  `id=str(segment_id)`, `confidence=segment["score"]`.

### `Engine/services/damage_model_service.py` — YOLO segmentation (Ultralytics)
Same caching pattern (`_MODEL`, `_DEVICE`, `_MODEL_READY`, `_MODEL_LOAD_ERROR`).

- `load_damage_model()` — raises `FileNotFoundError` if `DAMAGE_MODEL_PATH`
  missing; `YOLO(str(DAMAGE_MODEL_PATH))`.
- `run_damage_model(image)` — `model.predict(source=image, verbose=False,
  retina_masks=True)` — **no `conf=` kwarg** (see bug above).
  `retina_masks=True` gives full-resolution masks matching the input image.
- `postprocess_damage_raw_outputs(raw, image)` — trivial wrapper:
  `{"results": raw}`.
- `get_damage_id2label() -> {int: str}` — from `model.model.names`.
- `build_damage_detections(result) -> list[Detection]` — takes
  `results[0]`; bails if `.masks`/`.boxes` is `None`; per-instance mask
  threshold `mask.data[i] > 0.5`; `type="damage"`, `matches=None`.

### `Engine/services/combine_service.py` — `combine_results()`
The core of "combined" mode. **Output is damage-centric**: one `Detection`
per damage instance; car-part detections that don't overlap any damage are
never emitted standalone.

```python
def combine_results(damage_result, car_parts_result) -> list[Detection]:
    for damage in damage_segments:
        damage_area = damage_mask.sum()
        for part in car_part_segments:
            intersection = (damage_mask & part_mask).sum()
            if intersection == 0: continue
            score = intersection / damage_area                  # 0-1
            coveragePercent = intersection / part_area * 100     # 0-100
            matches.append(MatchInfo(part=part.label, score=..., coveragePercent=...))
        matches.sort(key=lambda m: m.score, reverse=True)  # best-covered part first
        combined_detections.append(Detection(..., matches=matches or None))
```
- **`score`** = what fraction of *this damage's* pixels fall on this part
  (normalized per-damage).
- **`coveragePercent`** = what percent of *this part's total area* is
  covered by this damage (normalized per-part, as a 0–100 number, not 0–1).
- Zero-area/`None`-polygon damages or parts are silently skipped (no
  `message` set on partial skips).
- Private helpers `_extract_damage_segments()` / `_extract_car_part_segments()`
  normalize each model's raw wrapper dict into a common
  `{id, label_id, label, confidence, mask}` shape.

---

## 4. React_Client — `React_Client/`

TypeScript + React 18 + Vite + Tailwind SPA. Ships two ways: `npm run dev`
for development, or packaged into a standalone Windows `.exe` via PyInstaller
(see [`React_Client/build.ps1`](React_Client/build.ps1)) for running on
machines without Node/a dev environment.

### Request flow (user's perspective)
1. `ImageUploader` — drag-drop or click-to-browse; only accepts
   `file.type.startsWith('image/')`.
2. `ModeSelector` — pick `damage_only` / `car_parts_only` / `combined`
   (default `combined`).
3. Click **Run analysis** → `useAnalyze().run(file, mode)`.
4. `analyzeImage()` ([`src/api/analyze.ts`](React_Client/src/api/analyze.ts)):
   reads the file as an `ArrayBuffer`, base64-encodes it manually
   (byte-by-byte `String.fromCharCode` + `btoa`), builds a
   `ClientAnalyzeRequest`-shaped JSON body (`requestId` via
   `crypto.randomUUID()`), and `POST`s it straight to
   `{apiUrl}/analyze` — **the API service directly, port 8002** (no
   intermediate Client hop).
5. Response is either `{status: "ok", detections: [...], image: {...}}` or
   `{status: "error", message: "..."}` — `useAnalyze` maps both, plus any
   network/HTTP failure, into a single `{ loading, result, error }` state.
6. `App.tsx` renders `ResultCanvas` (interactive polygon overlay on the
   original image) + `DetectionList` (grouped sidebar), sharing a
   `selectedId`/hover state so clicking either one highlights the other.

### Key files

**`src/App.tsx`** — top-level state/orchestration: `file`, `preview` (via
`URL.createObjectURL`), `mode`, `selectedId`, `heroVisible` (an
`IntersectionObserver` on the hero section toggles `Navbar`'s dark/light
styling). Renders `Navbar` → `HeroSection` → a two-column "console" (left:
3 numbered step cards for upload/mode/run; right: result panel) → footer.

**`src/api/analyze.ts`**
```ts
declare global { interface Window { __APP_CONFIG__?: { apiUrl?: string } } }
const API_BASE = window.__APP_CONFIG__?.apiUrl || 'http://localhost:8002'

export async function analyzeImage(file: File, mode: Mode): Promise<AnalyzeResponse>
```

**`src/hooks/useAnalyze.ts`**
```ts
export function useAnalyze(): { loading, result, error, run(file, mode) }
```
Distinguishes thrown network errors from a successful-HTTP-but-`status:
"error"` response — both end up in the same `error` string.

**`src/types/models.ts`** — TypeScript mirror of `models.py` (`Point`,
`Polygon`, `ImageInfo`, `MatchInfo`, `Detection`, `AnalyzeResponse`, and
`type Mode = 'combined' | 'car_parts_only' | 'damage_only'`).

**Components** (`src/components/`):
- `Navbar.tsx` — fixed nav, `dark: boolean` prop switches styling
  transparent-on-hero vs. frosted-light when scrolled past it.
- `HeroSection.tsx` — marketing hero with an inline SVG car illustration and
  a CTA that scrolls to the console section.
- `ImageUploader.tsx` — drag/drop + click-to-browse, MIME-type gated.
- `ModeSelector.tsx` — 3-button grid bound to `Mode`.
- `ResultCanvas.tsx` — the most complex component; a `<canvas>` that:
  - Draws the source image, then all detection polygons in 3 layers
    (normal → hovered → selected, so the active shape is always topmost).
  - `drawPoly()` fills each polygon with a semi-transparent color (hex alpha
    suffix trick, e.g. `'#3b82f6' + '45'`) and strokes it; highlighted
    shapes get a `shadowBlur` glow.
  - `drawLabel()` computes the polygon centroid and draws a rounded-rect
    pill with the capitalized label, font size scaling with canvas width.
  - `getCanvasCoords()` converts mouse `clientX/Y` to canvas-pixel space,
    correcting for the CSS-scaled display size vs. native pixel dimensions.
  - `handleMouseMove`/`handleClick` hit-test via a standard ray-casting
    `pointInPolygon()`, iterating detections in reverse (last-drawn wins).
  - Uses `useRef` mirrors of props/state so the `draw()` calls fired from
    async `<img>` `onload` callbacks always read fresh values.
- `DetectionList.tsx` — splits detections into "Damage"/"Car Parts"
  sections; each row shows a confidence badge (≥80% emerald, ≥50% amber,
  else slate) and, if `matches` is present, a nested list of
  `"{score×100}% overlap · {coveragePercent}% of part"`.

### Runtime config — one `config.json`, two consumers
The API endpoint is **not** baked in at build time (that's the old
`.env.local`/`VITE_API_URL` approach, now removed). Instead,
[`React_Client/config.json`](React_Client/config.json) is the single source
of truth:
```json
{ "apiUrl": "http://193.106.55.108:8002" }
```
- **`npm run dev` / `npm run preview`**: a custom Vite plugin in
  [`vite.config.ts`](React_Client/vite.config.ts) serves `/config.js` on
  every request by reading `config.json` fresh off disk.
- **The packaged `.exe`**: [`server/serve.py`](React_Client/server/serve.py)
  (a stdlib-only Python HTTP server, no Node/Electron needed at runtime)
  serves the built `dist/` assets and serves `/config.js` the same way, but
  reading `config.json` from **next to the executable** — so a deployed exe
  can be repointed at a different API server without rebuilding.
- Either way, `index.html` loads `<script src="/config.js">` before the
  React bundle, which sets `window.__APP_CONFIG__ = { apiUrl }`; that's what
  `analyze.ts` reads.
- [`build.ps1`](React_Client/build.ps1): `npm run build` → creates an
  isolated venv at `server/.venv` (kept separate from the ML services' venv)
  → installs PyInstaller there → packages `server/serve.py` with the built
  `dist/` bundled in via `--add-data` → copies `config.json` into
  `release/config.json` as the exe's starting config → outputs
  `React_Client/release/CarCalClient.exe` + `config.json`, ready to copy to
  any Windows machine.

---

## 5. Running everything locally

```powershell
# Python services (from repo root, with the venv activated)
uvicorn API.API_main:app --port 8002 --reload
uvicorn Engine.Engine_main:app --port 8001 --reload

# React client
cd React_Client
npm install
npm run dev
```
Or use the VS Code "Run All Services" compound launch config — note it
still tries to also launch `Client.Client_main:app` on port 8000, which will
fail to import now that `Client/` is removed; that leg needs updating or
removing from `.vscode/launch.json` if you rely on it.

`GET http://localhost:8002/services` (opened automatically by that launch
config's `open-dashboard` task) shows a live status dashboard — expect the
"Client" card to always read `CONNECTION_FAILED` since nothing serves port
8000 anymore.

Health checks: `GET /health` on API (8002) and Engine (8001);
`GET http://localhost:8001/health` on Engine returns the extended
`EngineHealthResponse` with per-model readiness.

---

## 6. Known rough edges (worth fixing eventually)

| # | File | Issue |
|---|------|-------|
| 1 | `API/API_methods.py` | `filename=` typo (should be `FileName=`) silently drops the uploaded filename between API and Engine. |
| 2 | `Engine/Engine_methods.py` | The `except` branch's `EngineAnalyzeResponse(...)` omits required field `mode=`, so it raises instead of returning the intended graceful error response — any processing exception becomes a raw 500. |
| 3 | `Engine/EngineConfig.py` / `damage_model_service.py` | `DAMAGE_MODEL_CONFIDENCE` is defined but never passed to `YOLO.predict()`; the damage model runs at Ultralytics' default threshold, not 0.25. |
| 4 | `Engine/EngineConfig.py` | `DEVICE` and `MAX_DETECTIONS` are declared but unused; `API/APIConfig.py`'s `ENGINE_HEALTH_URL` and `IMAGE_SIZE` are likewise unused. |
| 5 | `API/dashboard.py` | `run_pipeline_test()` is dead code — not wired to any route. |
| 6 | `.vscode/launch.json` / `API/dashboard.py` | Both still reference the removed `Client/` service on port 8000. |
| 7 | `CLAUDE.md` | Describes `Client/` as live and `React_Client/` as an empty placeholder — both are now inaccurate. |
| 8 | Root `.gitignore` | The Engine model-artifact ignore path (`Engine/Artifacts/Car_parts/...`) uses different casing than the real directory (`Engine/artifacts/car_parts/...`) — won't match on a case-sensitive filesystem. |
| 9 | `Client/Client_methods.py` (removed, but relevant if `Client/` is ever restored) | Used the synchronous `requests` library inside an `async def` handler, blocking the event loop during the API call despite being declared `async`. |
