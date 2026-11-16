"""Export OpenAPI schema for MemoryMesh REST API.

Usage:
    python scripts/export_openapi.py [output_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from memorymesh.api.main import create_app


def export_openapi(output_path: str | Path = "openapi.json") -> None:
    """Generate and write the OpenAPI schema as JSON."""
    app = create_app()
    schema = app.openapi()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"OpenAPI schema successfully written to {out}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    export_openapi(target)
