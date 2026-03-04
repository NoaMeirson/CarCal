import requests
from fastapi.responses import HTMLResponse


def check_service_health(url: str) -> str:
    try:
        response = requests.get(url, timeout=1)

        if response.status_code == 200:
            return "UP"

        return f"UNEXPECTED_STATUS_{response.status_code}"

    except requests.exceptions.Timeout:
        return "TIMEOUT"

    except requests.exceptions.ConnectionError:
        return "CONNECTION_FAILED"

    except Exception:
        return "ERROR"


def services_dashboard():

    client_status = check_service_health("http://localhost:8000/health")
    engine_status = check_service_health("http://localhost:8001/health")
    api_status = "UP"

    return HTMLResponse(f"""
    <html>
    <head>
        <title>CarCal Services Dashboard</title>
        <style>
            body {{ font-family: Arial; background:#f5f5f5; padding:40px;}}
            .container {{ display:flex; gap:20px;}}
            .card {{ background:white; padding:20px; border-radius:10px; width:250px;}}
            h1 {{ margin-bottom:30px;}}
            a {{ display:block; margin:5px 0;}}
            .status {{ font-weight:bold; margin-bottom:10px;}}
        </style>
    </head>

    <body>

    <h1>CarCal Microservices Dashboard</h1>

    <div class="container">

        <div class="card">
            <h2>Client</h2>
            <div class="status">Status: {client_status}</div>
            <a href="http://localhost:8000/docs" target="_blank">Swagger</a>
            <a href="http://localhost:8000/health" target="_blank">Health</a>
        </div>

        <div class="card">
            <h2>API</h2>
            <div class="status">Status: {api_status}</div>
            <a href="http://localhost:8002/docs" target="_blank">Swagger</a>
            <a href="http://localhost:8002/health" target="_blank">Health</a>
        </div>

        <div class="card">
            <h2>Engine</h2>
            <div class="status">Status: {engine_status}</div>
            <a href="http://localhost:8001/docs" target="_blank">Swagger</a>
            <a href="http://localhost:8001/health" target="_blank">Health</a>
        </div>

    </div>

    </body>
    </html>
    """)

import requests

def run_pipeline_test():

    report = []

    # check client
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            report.append("Client service reachable")
        else:
            report.append("Client health endpoint returned unexpected status")
            return report

    except:
        report.append("Client service not reachable. Check Client server on port 8000")
        return report


    # check engine
    try:
        r = requests.get("http://localhost:8001/health", timeout=2)
        if r.status_code == 200:
            report.append("Engine service reachable")
        else:
            report.append("Engine health endpoint returned unexpected status")
            return report

    except:
        report.append("Engine service not reachable. Check Engine server on port 8001")
        return report


    # test analyze pipeline
    try:

        payload = {
            "requestId": "pipeline_test",
            "imageBase64": "test"
        }

        r = requests.post(
            "http://localhost:8002/analyze",
            json=payload,
            timeout=3
        )

        if r.status_code == 200:
            report.append("API analyze pipeline working")
        else:
            report.append("Analyze endpoint returned error")
            return report

    except Exception as e:

        report.append("API could not process analyze request")
        report.append("Possible failure in API → Engine communication")

        return report


    report.append("SYSTEM STATUS: OK")

    return report