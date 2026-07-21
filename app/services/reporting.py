from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import REPORTS_DIR


class ReportingService:
    def create_report(self, payload: dict) -> dict:
        report_id = f"PHAA-{uuid4().hex[:10].upper()}"
        created_at = datetime.now(timezone.utc).isoformat()
        report = {
            "report_id": report_id,
            "created_at": created_at,
            **payload,
        }
        path = REPORTS_DIR / f"{report_id}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
