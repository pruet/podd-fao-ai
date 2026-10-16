import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from fastapi import HTTPException


@dataclass
class JobRecord:
    job_key: str               # composite: f"{tenant_key}:{integration_code}:{operation}:{event_id}"
    tenant_key: str
    integration_code: str
    event_id: str
    operation: str
    body_hash: str             # sha256 hex of raw webhook body
    status: str                # ACCEPTED, RUNNING, RESULT_READY, CALLBACK_PENDING, COMPLETED, FAILED, CANCELLED
    request_payload: Dict[str, Any]
    model_result: Optional[Dict[str, Any]] = None
    callback_payload: Optional[Dict[str, Any]] = None
    callback_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @staticmethod
    def compute_key(tenant_key: str, integration_code: str, operation: str, event_id: str) -> str:
        return f"{tenant_key}:{integration_code}:{operation}:{event_id}"

    @staticmethod
    def compute_body_hash(raw_body: bytes) -> str:
        return hashlib.sha256(raw_body).hexdigest()
