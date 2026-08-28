"""
ModelForge AI — Enterprise Application Entrypoint Runner
Launches the FastAPI backend server, Celery worker orchestration, and health telemetry.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))


def main():
    print("====================================================================")
    print("  🚀 ModelForge AI — Enterprise ML Lifecycle Platform Server")
    print("  Version: 1.0.0 | Environment: Production / Development")
    print("  Swagger UI: http://localhost:8000/api/v1/docs")
    print("  Frontend UI: http://localhost:3000")
    print("====================================================================")

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT", "development").lower() == "development",
        app_dir=str(backend_dir),
        log_level="info",
    )


if __name__ == "__main__":
    main()
