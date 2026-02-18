# Contracts

This folder defines the shared contracts between services.

## What’s included

- `contracts/models.py` – Pydantic models
- `contracts/schemas/*.schema.json` – Generated JSON Schemas
- `contracts/examples/*.json` – Example payloads

## Protocol suggestion

- **Client ↔ API:** REST (easy upload via multipart, browser tools)
- **API ↔ Engine:** gRPC (fast, schema-first, better for internal service-to-service)

These contracts are protocol-agnostic and can be used with REST everywhere if you prefer.

## REST endpoints suggestion (v1)

- `POST /v1/analyze` → `AnalyzeRequest` → `AnalyzeResponse`
- `GET /v1/health` → `HealthResponse`

Errors: return `ErrorResponse` with a proper HTTP status code.

## Regenerate schemas

```bash
python -m contracts.generate_schemas
```
