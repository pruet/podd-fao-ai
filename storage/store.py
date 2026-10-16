import os
import json
import sqlite3
import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from fastapi import HTTPException

from .jobs import JobRecord

IS_GAE = os.getenv("GAE_ENV") == "standard" or "GAE_SERVICE" in os.environ
REQUIRE_SHARED_STORE = os.getenv("REQUIRE_SHARED_STORE", "false").lower() == "true" or IS_GAE
USE_DATASTORE = IS_GAE or os.getenv("USE_DATASTORE") == "true"

DB_PATH = "/tmp/api_logs.db" if IS_GAE else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api_logs.db")
LOGS_DIR = "/tmp/logs" if IS_GAE else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

datastore_client = None

def get_datastore_client():
    global datastore_client, USE_DATASTORE
    if datastore_client is None and USE_DATASTORE:
        try:
            from google.cloud import datastore
            datastore_client = datastore.Client()
        except Exception as e:
            print(f"[STORAGE ERROR] Failed to initialize Datastore client: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(
                    status_code=503,
                    detail="Shared Datastore is required in this deployment environment but could not be initialized."
                )
            USE_DATASTORE = False
    return datastore_client


def init_sqlite_db(db_path: str = DB_PATH):
    """Initializes and runs migrations on SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. API Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_key TEXT DEFAULT 'standalone',
                timestamp TEXT,
                method TEXT,
                path TEXT,
                status_code INTEGER,
                latency REAL,
                client_ip TEXT,
                request_params TEXT,
                response_body TEXT,
                steps_json TEXT,
                image_path TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0
            )
        """)

        # 2. Cluster Reports Table with tenant isolation and unique upsert
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cluster_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_key TEXT NOT NULL DEFAULT 'unassigned_legacy',
                deployment_id TEXT NOT NULL DEFAULT 'lahis',
                tenant_schema TEXT NOT NULL DEFAULT 'legacy',
                timestamp TEXT,
                date_str TEXT,
                report_id TEXT,
                village_id TEXT,
                village_name TEXT,
                raw_payload TEXT,
                detected_disease TEXT
            )
        """)

        # 3. Durable Job Records Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_key TEXT UNIQUE NOT NULL,
                tenant_key TEXT NOT NULL,
                integration_code TEXT NOT NULL,
                event_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                request_payload TEXT,
                model_result TEXT,
                callback_payload TEXT,
                callback_response TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Check existing columns and run migrations
        cursor.execute("PRAGMA table_info(api_logs)")
        log_columns = [row[1] for row in cursor.fetchall()]
        if "tenant_key" not in log_columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN tenant_key TEXT DEFAULT 'standalone'")
        if "image_path" not in log_columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN image_path TEXT")
        if "prompt_tokens" not in log_columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN prompt_tokens INTEGER DEFAULT 0")
        if "output_tokens" not in log_columns:
            cursor.execute("ALTER TABLE api_logs ADD COLUMN output_tokens INTEGER DEFAULT 0")

        cursor.execute("PRAGMA table_info(cluster_reports)")
        cluster_columns = [row[1] for row in cursor.fetchall()]
        if "tenant_key" not in cluster_columns:
            cursor.execute("ALTER TABLE cluster_reports ADD COLUMN tenant_key TEXT NOT NULL DEFAULT 'unassigned_legacy'")
        if "deployment_id" not in cluster_columns:
            cursor.execute("ALTER TABLE cluster_reports ADD COLUMN deployment_id TEXT NOT NULL DEFAULT 'lahis'")
        if "tenant_schema" not in cluster_columns:
            cursor.execute("ALTER TABLE cluster_reports ADD COLUMN tenant_schema TEXT NOT NULL DEFAULT 'legacy'")
        if "detected_disease" not in cluster_columns:
            cursor.execute("ALTER TABLE cluster_reports ADD COLUMN detected_disease TEXT")

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_tenant ON api_logs(tenant_key, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_tenant_village ON cluster_reports(tenant_key, village_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_tenant_date ON cluster_reports(tenant_key, date_str)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_tenant_report ON cluster_reports(tenant_key, report_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON job_records(status)")

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing SQLite DB: {e}")


# Initialize SQLite DB on load
init_sqlite_db()


def record_cluster_report(
    report_id: str,
    village_id: str,
    village_name: Optional[str] = None,
    date_str: Optional[str] = None,
    raw_payload: str = "",
    detected_disease: Optional[str] = None,
    tenant_key: Optional[str] = None,
    deployment_id: Optional[str] = None,
    tenant_schema: Optional[str] = None
):
    """
    Idempotently records a report into cluster storage scoped to tenant_key.
    Upserts by (tenant_key, report_id) so duplicate webhooks never inflate counts.
    """
    actual_tenant_key = str(tenant_key or "standalone")
    actual_deployment_id = str(deployment_id or "lahis")
    actual_tenant_schema = str(tenant_schema or "default")
    actual_date_str = str(date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    timestamp = datetime.now(timezone.utc).isoformat()
    client = get_datastore_client()

    if USE_DATASTORE and client:
        try:
            from google.cloud import datastore
            # Entity key name incorporates tenant_key and report_id for atomic upsert
            key = client.key("ClusterReport", f"{actual_tenant_key}:{report_id}")
            entity = datastore.Entity(key=key, exclude_from_indexes=("raw_payload",))
            entity.update({
                "tenant_key": actual_tenant_key,
                "deployment_id": actual_deployment_id,
                "tenant_schema": actual_tenant_schema,
                "timestamp": timestamp,
                "date_str": actual_date_str,
                "report_id": str(report_id),
                "village_id": str(village_id),
                "village_name": str(village_name or ""),
                "raw_payload": raw_payload,
                "detected_disease": str(detected_disease or "")
            })
            client.put(entity)
            return
        except Exception as e:
            print(f"[DATASTORE ERROR] Error recording ClusterReport: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(status_code=503, detail="Shared datastore failed to record cluster report.")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Check if record exists for this tenant and report_id
        cursor.execute(
            "SELECT id FROM cluster_reports WHERE tenant_key = ? AND report_id = ?",
            (actual_tenant_key, str(report_id))
        )
        existing = cursor.fetchone()
        if existing:
            cursor.execute("""
                UPDATE cluster_reports
                SET timestamp = ?, date_str = ?, village_id = ?, village_name = ?, raw_payload = ?, detected_disease = ?
                WHERE id = ?
            """, (timestamp, actual_date_str, str(village_id), str(village_name or ""), raw_payload, str(detected_disease or ""), existing[0]))
        else:
            cursor.execute("""
                INSERT INTO cluster_reports (
                    tenant_key, deployment_id, tenant_schema, timestamp, date_str, report_id, village_id, village_name, raw_payload, detected_disease
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (actual_tenant_key, actual_deployment_id, actual_tenant_schema, timestamp, actual_date_str, str(report_id), str(village_id), str(village_name or ""), raw_payload, str(detected_disease or "")))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLITE ERROR] Error saving cluster report to SQLite: {e}")


def get_village_reports_for_date(village_id: str, date_str: str, tenant_key: Optional[str] = None) -> List[dict]:
    """Retrieves all reports for a specific village on a date, scoped to tenant_key if provided."""
    client = get_datastore_client()
    if USE_DATASTORE and client:
        try:
            query = client.query(kind="ClusterReport")
            if tenant_key:
                query.add_filter("tenant_key", "=", str(tenant_key))
            query.add_filter("village_id", "=", str(village_id))
            query.add_filter("date_str", "=", str(date_str))
            results = list(query.fetch())
            return [
                {
                    "id": str(entity.key.id_or_name),
                    "tenant_key": entity.get("tenant_key"),
                    "timestamp": entity.get("timestamp"),
                    "date_str": entity.get("date_str"),
                    "report_id": entity.get("report_id"),
                    "village_id": entity.get("village_id"),
                    "village_name": entity.get("village_name"),
                    "raw_payload": entity.get("raw_payload"),
                    "detected_disease": entity.get("detected_disease", "")
                }
                for entity in results
            ]
        except Exception as e:
            print(f"[DATASTORE ERROR] Error querying ClusterReport from Datastore: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(status_code=503, detail="Shared datastore is unavailable.")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if tenant_key:
            cursor.execute(
                "SELECT * FROM cluster_reports WHERE tenant_key = ? AND village_id = ? AND date_str = ?",
                (str(tenant_key), str(village_id), str(date_str))
            )
        else:
            cursor.execute(
                "SELECT * FROM cluster_reports WHERE village_id = ? AND date_str = ?",
                (str(village_id), str(date_str))
            )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "tenant_key": row["tenant_key"],
                "timestamp": row["timestamp"],
                "date_str": row["date_str"],
                "report_id": row["report_id"],
                "village_id": row["village_id"],
                "village_name": row["village_name"],
                "raw_payload": row["raw_payload"],
                "detected_disease": row["detected_disease"] if "detected_disease" in row.keys() else ""
            }
            for row in rows
        ]
    except Exception as e:
        print(f"[SQLITE ERROR] Error querying cluster reports: {e}")
        return []


def get_village_all_reports(village_id: str, tenant_key: Optional[str] = None) -> List[dict]:
    """Retrieves all reports for a specific village across history, scoped to tenant_key if provided."""
    client = get_datastore_client()
    if USE_DATASTORE and client:
        try:
            query = client.query(kind="ClusterReport")
            if tenant_key:
                query.add_filter("tenant_key", "=", str(tenant_key))
            query.add_filter("village_id", "=", str(village_id))
            results = list(query.fetch())
            return [
                {
                    "id": str(entity.key.id_or_name),
                    "tenant_key": entity.get("tenant_key"),
                    "timestamp": entity.get("timestamp"),
                    "date_str": entity.get("date_str"),
                    "report_id": entity.get("report_id"),
                    "village_id": entity.get("village_id"),
                    "village_name": entity.get("village_name"),
                    "raw_payload": entity.get("raw_payload"),
                    "detected_disease": entity.get("detected_disease", "")
                }
                for entity in results
            ]
        except Exception as e:
            print(f"[DATASTORE ERROR] Error querying ClusterReport from Datastore: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(status_code=503, detail="Shared datastore is unavailable.")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if tenant_key:
            cursor.execute(
                "SELECT * FROM cluster_reports WHERE tenant_key = ? AND village_id = ?",
                (str(tenant_key), str(village_id))
            )
        else:
            cursor.execute(
                "SELECT * FROM cluster_reports WHERE village_id = ?",
                (str(village_id),)
            )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "tenant_key": row["tenant_key"],
                "timestamp": row["timestamp"],
                "date_str": row["date_str"],
                "report_id": row["report_id"],
                "village_id": row["village_id"],
                "village_name": row["village_name"],
                "raw_payload": row["raw_payload"],
                "detected_disease": row["detected_disease"] if "detected_disease" in row.keys() else ""
            }
            for row in rows
        ]
    except Exception as e:
        print(f"[SQLITE ERROR] Error querying cluster reports: {e}")
        return []


def log_request_response(
    method: str,
    path: str,
    status_code: int,
    latency: float,
    client_ip: str,
    request_params: dict,
    response_body: str,
    steps_json: Optional[dict] = None,
    image_path: Optional[str] = None,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    tenant_key: Optional[str] = None
):
    """Logs API requests and responses scoped to tenant_key, sanitizing any sensitive credentials."""
    def sanitize(item):
        if isinstance(item, dict):
            clean = {}
            for k, v in item.items():
                k_lower = str(k).lower()
                if "secret" in k_lower or "token" in k_lower or "authorization" in k_lower or "key" in k_lower:
                    clean[k] = "[REDACTED]"
                else:
                    clean[k] = sanitize(v)
            return clean
        elif isinstance(item, list):
            return [sanitize(i) for i in item]
        elif isinstance(item, (str, int, float, bool, type(None))):
            return item
        else:
            return str(item)

    clean_params = sanitize(request_params)
    clean_steps = sanitize(steps_json) if steps_json else None
    timestamp = datetime.now(timezone.utc).isoformat()

    actual_tenant_key = str(tenant_key or "standalone")
    client = get_datastore_client()
    if USE_DATASTORE and client:
        try:
            from google.cloud import datastore
            key = client.key("ApiLog")
            entity = datastore.Entity(
                key=key,
                exclude_from_indexes=("response_body", "steps_json", "request_params", "image_path")
            )
            entity.update({
                "tenant_key": actual_tenant_key,
                "timestamp": timestamp,
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency": latency,
                "client_ip": client_ip,
                "request_params": json.dumps(clean_params, ensure_ascii=False),
                "response_body": response_body,
                "steps_json": json.dumps(clean_steps, ensure_ascii=False) if clean_steps else None,
                "image_path": image_path,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens
            })
            client.put(entity)
            return
        except Exception as e:
            print(f"[DATASTORE ERROR] Error logging to Datastore: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(status_code=503, detail="Shared datastore is unavailable.")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_logs (
                tenant_key, timestamp, method, path, status_code, latency, client_ip, request_params, response_body, steps_json, image_path, prompt_tokens, output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            actual_tenant_key,
            timestamp,
            method,
            path,
            status_code,
            latency,
            client_ip,
            json.dumps(clean_params, ensure_ascii=False),
            response_body,
            json.dumps(clean_steps, ensure_ascii=False) if clean_steps else None,
            image_path,
            prompt_tokens,
            output_tokens
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLITE ERROR] Error logging to SQLite: {e}")


def get_or_create_job(
    tenant_key: str,
    integration_code: str,
    operation: str,
    event_id: str,
    raw_body_bytes: bytes,
    request_payload: Dict[str, Any]
) -> Tuple[JobRecord, str]:
    """
    Durable job deduplication ledger.
    Returns (job, 'created' | 'replay' | 'conflict').
    - 'created': Job is newly inserted with status ACCEPTED.
    - 'replay': Duplicate request with same body hash.
    - 'conflict': Duplicate request key but DIFFERENT body hash (raises 409).
    """
    job_key = JobRecord.compute_key(tenant_key, integration_code, operation, event_id)
    body_hash = JobRecord.compute_body_hash(raw_body_bytes)
    now_iso = datetime.now(timezone.utc).isoformat()

    client = get_datastore_client()
    if USE_DATASTORE and client:
        try:
            from google.cloud import datastore
            key = client.key("JobRecord", job_key)
            entity = client.get(key)
            if entity:
                stored_hash = entity.get("body_hash")
                if stored_hash != body_hash:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Conflict: Event {event_id} already exists with different payload body hash"
                    )
                job = JobRecord(
                    job_key=job_key,
                    tenant_key=entity.get("tenant_key"),
                    integration_code=entity.get("integration_code"),
                    event_id=entity.get("event_id"),
                    operation=entity.get("operation"),
                    body_hash=stored_hash,
                    status=entity.get("status"),
                    request_payload=json.loads(entity.get("request_payload") or "{}"),
                    model_result=json.loads(entity.get("model_result") or "null"),
                    callback_payload=json.loads(entity.get("callback_payload") or "null"),
                    callback_response=json.loads(entity.get("callback_response") or "null"),
                    error_message=entity.get("error_message"),
                    created_at=entity.get("created_at"),
                    updated_at=entity.get("updated_at")
                )
                return job, "replay"

            # Create new entity
            new_entity = datastore.Entity(key=key, exclude_from_indexes=("request_payload", "model_result", "callback_payload", "callback_response"))
            new_entity.update({
                "tenant_key": str(tenant_key),
                "integration_code": str(integration_code),
                "operation": str(operation),
                "event_id": str(event_id),
                "body_hash": str(body_hash),
                "status": "ACCEPTED",
                "request_payload": json.dumps(request_payload, ensure_ascii=False),
                "model_result": None,
                "callback_payload": None,
                "callback_response": None,
                "error_message": None,
                "created_at": now_iso,
                "updated_at": now_iso
            })
            client.put(new_entity)
            return JobRecord(
                job_key=job_key,
                tenant_key=tenant_key,
                integration_code=integration_code,
                event_id=event_id,
                operation=operation,
                body_hash=body_hash,
                status="ACCEPTED",
                request_payload=request_payload,
                created_at=now_iso,
                updated_at=now_iso
            ), "created"
        except HTTPException:
            raise
        except Exception as e:
            print(f"[DATASTORE ERROR] Error in get_or_create_job: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(status_code=503, detail="Shared datastore is unavailable for job persistence.")

    # SQLite Ledger
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM job_records WHERE job_key = ?", (job_key,))
    row = cursor.fetchone()
    if row:
        stored_hash = row["body_hash"]
        if stored_hash != body_hash:
            conn.close()
            raise HTTPException(
                status_code=409,
                detail=f"Conflict: Event {event_id} already exists with different payload body hash"
            )
        job = JobRecord(
            job_key=job_key,
            tenant_key=row["tenant_key"],
            integration_code=row["integration_code"],
            event_id=row["event_id"],
            operation=row["operation"],
            body_hash=stored_hash,
            status=row["status"],
            request_payload=json.loads(row["request_payload"] or "{}"),
            model_result=json.loads(row["model_result"]) if row["model_result"] else None,
            callback_payload=json.loads(row["callback_payload"]) if row["callback_payload"] else None,
            callback_response=json.loads(row["callback_response"]) if row["callback_response"] else None,
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
        conn.close()
        return job, "replay"

    cursor.execute("""
        INSERT INTO job_records (
            job_key, tenant_key, integration_code, event_id, operation, body_hash, status, request_payload, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_key,
        tenant_key,
        integration_code,
        event_id,
        operation,
        body_hash,
        "ACCEPTED",
        json.dumps(request_payload, ensure_ascii=False),
        now_iso,
        now_iso
    ))
    conn.commit()
    conn.close()

    return JobRecord(
        job_key=job_key,
        tenant_key=tenant_key,
        integration_code=integration_code,
        event_id=event_id,
        operation=operation,
        body_hash=body_hash,
        status="ACCEPTED",
        request_payload=request_payload,
        created_at=now_iso,
        updated_at=now_iso
    ), "created"


def update_job(
    job_key: str,
    status: Optional[str] = None,
    model_result: Optional[Dict[str, Any]] = None,
    callback_payload: Optional[Dict[str, Any]] = None,
    callback_response: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
):
    """Updates job progress and status atomically."""
    now_iso = datetime.now(timezone.utc).isoformat()
    client = get_datastore_client()
    if USE_DATASTORE and client:
        try:
            key = client.key("JobRecord", job_key)
            entity = client.get(key)
            if entity:
                if status:
                    entity["status"] = status
                if model_result is not None:
                    entity["model_result"] = json.dumps(model_result, ensure_ascii=False)
                if callback_payload is not None:
                    entity["callback_payload"] = json.dumps(callback_payload, ensure_ascii=False)
                if callback_response is not None:
                    entity["callback_response"] = json.dumps(callback_response, ensure_ascii=False)
                if error_message is not None:
                    entity["error_message"] = str(error_message)
                entity["updated_at"] = now_iso
                client.put(entity)
                return
        except Exception as e:
            print(f"[DATASTORE ERROR] Error updating job in Datastore: {e}")
            if REQUIRE_SHARED_STORE:
                raise HTTPException(status_code=503, detail="Shared datastore is unavailable.")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        updates = ["updated_at = ?"]
        params = [now_iso]

        if status:
            updates.append("status = ?")
            params.append(status)
        if model_result is not None:
            updates.append("model_result = ?")
            params.append(json.dumps(model_result, ensure_ascii=False))
        if callback_payload is not None:
            updates.append("callback_payload = ?")
            params.append(json.dumps(callback_payload, ensure_ascii=False))
        if callback_response is not None:
            updates.append("callback_response = ?")
            params.append(json.dumps(callback_response, ensure_ascii=False))
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(str(error_message))

        params.append(job_key)
        sql = f"UPDATE job_records SET {', '.join(updates)} WHERE job_key = ?"
        cursor.execute(sql, tuple(params))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQLITE ERROR] Error updating job in SQLite: {e}")
