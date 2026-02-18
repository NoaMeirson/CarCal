from __future__ import annotations
"""
Generate JSON Schemas from the Pydantic models.

Run:
  python -m contracts.generate_schemas
"""
import json
from pathlib import Path
from pydantic import TypeAdapter
from . import models

OUT = Path(__file__).resolve().parent / "schemas"

TARGETS = {
  "AnalyzeRequest": models.AnalyzeRequest,
  "AnalyzeResponse": models.AnalyzeResponse,
  "EngineAnalyzeRequest": models.EngineAnalyzeRequest,
  "EngineAnalyzeResponse": models.EngineAnalyzeResponse,
  "ErrorResponse": models.ErrorResponse,
  "HealthResponse": models.HealthResponse,
}

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, typ in TARGETS.items():
        schema = TypeAdapter(typ).json_schema()
        (OUT / f"{name}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {len(TARGETS)} schemas to {OUT}")

if __name__ == "__main__":
    main()
