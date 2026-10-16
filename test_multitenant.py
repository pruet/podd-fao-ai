import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import io
import sqlite3
import json
import sys
import os
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
import httpx
from fastapi import HTTPException

# Ensure Src is in sys.path
sys.path.append(os.path.dirname(__file__))

from main import app, init_db, DB_PATH
from tenant import (
    TenantContext,
    TenantSigningKey,
    TenantRegistry,
    TenantTokenManager,
    LahisTenantClient,
    verify_hmac_signature,
    validate_timestamp_freshness,
    create_default_registry
)
from storage import (
    JobRecord,
    get_or_create_job,
    update_job,
    record_cluster_report,
    get_village_all_reports
)


def create_test_tenant_registry() -> TenantRegistry:
    """Creates an isolated registry with demo and tot tenants with distinct credentials."""
    registry = TenantRegistry()

    # demo tenant -> maps to tot.api.lahis.ohtk.org (hostname inversion)
    demo_key_v1 = TenantSigningKey(key_id="key-demo-1", version="1", secret="secret-demo-v1")
    demo_key_v2 = TenantSigningKey(key_id="key-demo-2", version="2", secret="secret-demo-v2")
    registry.register(TenantContext(
        tenant_key="lahis:demo",
        deployment_id="lahis",
        schema="demo",
        code="demo",
        name="Demo Tenant",
        api_base_url="https://tot.api.lahis.ohtk.org",
        integration_code="podd-ai-demo-v1",
        oauth_client_id="oauth-demo-client-id",
        oauth_client_secret="oauth-demo-client-secret",
        credential_version="v1",
        signing_keys={"key-demo-1": demo_key_v1, "key-demo-2": demo_key_v2},
        active_signing_key_id="key-demo-1",
        lang="lo",
        enabled_operations=("report.submitted", "detect-cluster", "ai.evaluation_requested")
    ))

    # tot tenant -> maps to demo.api.lahis.ohtk.org (hostname inversion)
    tot_key_v1 = TenantSigningKey(key_id="key-tot-1", version="1", secret="secret-tot-v1")
    tot_key_webhook = TenantSigningKey(key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET", version="1", secret="secret-tot-v1")
    registry.register(TenantContext(
        tenant_key="lahis:tot",
        deployment_id="lahis",
        schema="tot",
        code="tot",
        name="TOT Tenant",
        api_base_url="https://demo.api.lahis.ohtk.org",
        integration_code="podd-ai-tot-v1",
        oauth_client_id="oauth-tot-client-id",
        oauth_client_secret="oauth-tot-client-secret",
        credential_version="v1",
        signing_keys={"key-tot-1": tot_key_v1, "INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET": tot_key_webhook},
        active_signing_key_id="key-tot-1",
        lang="th",
        enabled_operations=("report.submitted", "detect-cluster", "ai.evaluation_requested")
    ))

    return registry


def make_signed_headers(
    method: str,
    path: str,
    raw_body: bytes,
    secret: str,
    schema: str,
    event_id: str,
    integration_code: str,
    key_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    signature: Optional[str] = None,
    alg: str = "HMAC-SHA256"
) -> Dict[str, str]:
    """Generates valid webhook headers according to the 7-step security specification."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    if signature is None:
        message = f"{method.upper()}\n{path}\n{ts}\n".encode("utf-8") + raw_body
        signature = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-OHTK-Tenant": schema,
        "X-OHTK-Event-ID": event_id,
        "X-OHTK-Integration": integration_code,
        "X-OHTK-Timestamp": ts,
        "X-OHTK-Signature": signature,
        "X-OHTK-Signature-Alg": alg
    }
    if key_id:
        headers["X-OHTK-Signing-Key-Id"] = key_id
    return headers


class TestMultiTenantLahis(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        init_db()
        self.client = TestClient(app)
        self.registry = create_test_tenant_registry()
        # Clean test tables
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute("DELETE FROM cluster_reports")
            conn.cursor().execute("DELETE FROM job_records")
            conn.cursor().execute("DELETE FROM api_logs")
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _mock_gemini_response(self, animal_type="Cattle", disease_name="Anthrax", confidence=0.95):
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50)
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": animal_type,
            "diseases": [
                {
                    "name": disease_name,
                    "confidence": confidence,
                    "reasoning": f"Symptoms indicative of {disease_name}"
                }
            ]
        })
        mock_genai_client.models.generate_content.return_value = mock_response
        return mock_genai_client

    # =========================================================================
    # SCENARIO 1: Concurrent Jobs A/B Isolation
    # =========================================================================
    @patch("main.tenant_registry")
    @patch("main.get_genai_client")
    async def test_scenario_01_concurrent_jobs_ab_isolation(self, mock_genai, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        mock_genai.return_value = self._mock_gemini_response()

        tenant_a = self.registry.get_by_schema("demo")
        tenant_b = self.registry.get_by_schema("tot")

        payload_a = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-tenant-a-1",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant_a.api_base_url},
            "report": {"id": "rep-tenant-a-101", "incidentDate": "2026-10-16"}
        }
        body_a = json.dumps(payload_a).encode("utf-8")
        headers_a = make_signed_headers("POST", "/analyze", body_a, "secret-demo-v1", "demo", "evt-tenant-a-1", "podd-ai-demo-v1")

        payload_b = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-tenant-b-1",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "tot", "apiBaseUrl": tenant_b.api_base_url},
            "report": {"id": "rep-tenant-b-201", "incidentDate": "2026-10-16"}
        }
        body_b = json.dumps(payload_b).encode("utf-8")
        headers_b = make_signed_headers("POST", "/analyze", body_b, "secret-tot-v1", "tot", "evt-tenant-b-1", "podd-ai-tot-v1")

        # Mock outbound requests per tenant
        with patch.object(LahisTenantClient, "fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch_imgs, \
             patch.object(LahisTenantClient, "submit_comment", new_callable=AsyncMock) as mock_comment, \
             patch.object(LahisTenantClient, "submit_cluster", new_callable=AsyncMock) as mock_cluster, \
             patch("main.Image.open") as mock_img:

            mock_fetch_imgs.return_value = [b"mock_img_bytes"]
            mock_comment.return_value = {"status": "accepted"}
            mock_cluster.return_value = {"status": "accepted", "id": "cluster-1"}
            mock_img.return_value = MagicMock()

            # Execute Request A
            res_a = self.client.post("/analyze", content=body_a, headers=headers_a)
            self.assertEqual(res_a.status_code, 200)

            # Execute Request B
            res_b = self.client.post("/analyze", content=body_b, headers=headers_b)
            self.assertEqual(res_b.status_code, 200)

            # Verify jobs in DB are partitioned under different tenant_keys
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.cursor().execute("SELECT tenant_key, event_id, status FROM job_records ORDER BY id ASC").fetchall()
            conn.close()

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["tenant_key"], "lahis:demo")
            self.assertEqual(rows[0]["event_id"], "evt-tenant-a-1")
            self.assertEqual(rows[0]["status"], "COMPLETED")

            self.assertEqual(rows[1]["tenant_key"], "lahis:tot")
            self.assertEqual(rows[1]["event_id"], "evt-tenant-b-1")
            self.assertEqual(rows[1]["status"], "COMPLETED")

    # =========================================================================
    # SCENARIO 2: Schema Does Not Match Hostname
    # =========================================================================
    @patch("main.tenant_registry")
    def test_scenario_02_schema_does_not_match_hostname(self, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant_tot = self.registry.get_by_schema("tot")

        # 1. Verify schema 'tot' maps to 'demo.api.lahis.ohtk.org', NOT 'tot.api.lahis.ohtk.org'
        self.assertEqual(tenant_tot.api_base_url, "https://demo.api.lahis.ohtk.org")

        # 2. Reject untrusted / malicious apiBaseUrl
        malicious_payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-malicious-1",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "tot", "apiBaseUrl": "https://malicious.attacker.org"},
            "report": {"id": "rep-malicious", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(malicious_payload).encode("utf-8")
        headers = make_signed_headers("POST", "/analyze", body, "secret-tot-v1", "tot", "evt-malicious-1", "podd-ai-tot-v1")

        response = self.client.post("/analyze", content=body, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("does not match approved registry URL", response.json()["detail"])

    # =========================================================================
    # SCENARIO 3: Invalid or Missing Signature / Timestamp / Key
    # =========================================================================
    @patch("main.tenant_registry")
    def test_scenario_03_invalid_missing_signature_timestamp_key(self, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant = self.registry.get_by_schema("demo")

        payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-sec-test",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-sec", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")

        # 1. Missing signature header -> 401
        h1 = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-sec-test", "podd-ai-demo-v1")
        del h1["X-OHTK-Signature"]
        r1 = self.client.post("/analyze", content=body, headers=h1)
        self.assertEqual(r1.status_code, 401)

        # 2. Bad HMAC signature -> 401
        h2 = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-sec-test", "podd-ai-demo-v1")
        h2["X-OHTK-Signature"] = "0000000000000000000000000000000000000000000000000000000000000000"
        r2 = self.client.post("/analyze", content=body, headers=h2)
        self.assertEqual(r2.status_code, 401)

        # 3. Drifted timestamp (> 5 minutes ago) -> 401
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        h3 = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-sec-test", "podd-ai-demo-v1", timestamp=old_ts)
        r3 = self.client.post("/analyze", content=body, headers=h3)
        self.assertEqual(r3.status_code, 401)
        self.assertIn("drift exceeds tolerance", r3.json()["detail"])

        # 4. Unknown Key ID -> 401
        h4 = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-sec-test", "podd-ai-demo-v1", key_id="unknown-key-999")
        r4 = self.client.post("/analyze", content=body, headers=h4)
        self.assertEqual(r4.status_code, 401)
        self.assertIn("Unknown signing key ID", r4.json()["detail"])

    # =========================================================================
    # SCENARIO 4: Key Cross-Contamination
    # =========================================================================
    @patch("main.tenant_registry")
    def test_scenario_04_key_cross_contamination(self, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant_tot = self.registry.get_by_schema("tot")

        payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-tot-spoof",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "tot", "apiBaseUrl": tenant_tot.api_base_url},
            "report": {"id": "rep-tot-spoof", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")

        # Sign Tenant 'tot' payload using Tenant 'demo' secret
        headers = make_signed_headers(
            method="POST",
            path="/analyze",
            raw_body=body,
            secret="secret-demo-v1",  # INCORRECT KEY FOR TOT
            schema="tot",
            event_id="evt-tot-spoof",
            integration_code="podd-ai-tot-v1"
        )
        response = self.client.post("/analyze", content=body, headers=headers)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid webhook signature", response.json()["detail"])

    # =========================================================================
    # SCENARIO 5: SSRF / Cross-Origin Redirect Containment
    # =========================================================================
    async def test_scenario_05_ssrf_and_cross_origin_containment(self):
        tenant = self.registry.get_by_schema("demo")
        token_mgr = TenantTokenManager()

        # 1. Verify URL targeting disallowed endpoint or external host raises HTTPException(400)
        client = LahisTenantClient(tenant, token_mgr)
        with self.assertRaises(HTTPException) as ctx:
            client._validate_safe_url("https://evil.attacker.org/steal-token")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Security violation: Target URL origin", ctx.exception.detail)

        with self.assertRaises(HTTPException) as ctx:
            client._validate_safe_url(f"{tenant.api_base_url}/admin/secret")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Disallowed endpoint path", ctx.exception.detail)

        # 2. Verify redirect to cross-origin is blocked
        mock_transport = httpx.MockTransport(
            lambda req: httpx.Response(302, headers={"Location": "https://evil.attacker.org/capture"})
        )
        async with httpx.AsyncClient(transport=mock_transport) as test_http_client:
            bound_client = LahisTenantClient(tenant, token_mgr, client=test_http_client)
            with patch.object(token_mgr, "get_token", new_callable=AsyncMock) as mock_get_token:
                mock_get_token.return_value = "safe_bearer_token"
                with self.assertRaises(HTTPException) as ctx:
                    await bound_client._request_with_retry("GET", f"{tenant.api_base_url}/api/integrations/v1/incidents/123")
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn("Security violation", ctx.exception.detail)

    # =========================================================================
    # SCENARIO 6: Village / Report ID Collision Under Different Tenants
    # =========================================================================
    @patch("main.tenant_registry")
    def test_scenario_06_village_report_id_collision_isolated(self, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema

        # Insert same report_id and village_id for two different tenants
        record_cluster_report(
            report_id="report-shared-id",
            village_id="village-v01",
            village_name="Village Alpha Demo",
            date_str="2026-10-16",
            raw_payload="{}",
            detected_disease="Anthrax",
            tenant_key="lahis:demo"
        )

        record_cluster_report(
            report_id="report-shared-id",
            village_id="village-v01",
            village_name="Village Alpha TOT",
            date_str="2026-10-16",
            raw_payload="{}",
            detected_disease="African Swine Fever",
            tenant_key="lahis:tot"
        )

        # Check tenant isolation in get_village_all_reports
        reports_demo = get_village_all_reports("village-v01", tenant_key="lahis:demo")
        reports_tot = get_village_all_reports("village-v01", tenant_key="lahis:tot")

        self.assertEqual(len(reports_demo), 1)
        self.assertEqual(reports_demo[0]["detected_disease"], "Anthrax")
        self.assertEqual(reports_demo[0]["village_name"], "Village Alpha Demo")

        self.assertEqual(len(reports_tot), 1)
        self.assertEqual(reports_tot[0]["detected_disease"], "African Swine Fever")
        self.assertEqual(reports_tot[0]["village_name"], "Village Alpha TOT")

        # Check clusters API grouping by (tenant_key, village_id)
        auth_headers = {"Authorization": "Basic YWRtaW46UG9kZEZhb1NlY3VyZTIwMjYh"}
        res = self.client.get("/api/clusters", headers=auth_headers)
        self.assertEqual(res.status_code, 200)
        clusters = res.json()["clusters"]
        self.assertEqual(len(clusters), 2)
        group_keys = [c["group_key"] for c in clusters]
        self.assertIn("lahis:demo:village-v01", group_keys)
        self.assertIn("lahis:tot:village-v01", group_keys)

    # =========================================================================
    # SCENARIO 7: Concurrent Duplicate Requests and Worker Restart (Idempotency)
    # =========================================================================
    @patch("main.tenant_registry")
    @patch("main.get_genai_client")
    def test_scenario_07_concurrent_duplicate_and_worker_restart(self, mock_genai, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        mock_genai.return_value = self._mock_gemini_response()
        tenant = self.registry.get_by_schema("demo")

        payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-dup-test",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-dup-test", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-dup-test", "podd-ai-demo-v1")

        with patch.object(LahisTenantClient, "fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch_imgs, \
             patch.object(LahisTenantClient, "submit_comment", new_callable=AsyncMock) as mock_comment, \
             patch.object(LahisTenantClient, "submit_cluster", new_callable=AsyncMock) as mock_cluster, \
             patch("main.Image.open") as mock_img:

            mock_fetch_imgs.return_value = [b"mock_img"]
            mock_comment.return_value = {"status": "ok"}
            mock_cluster.return_value = {"status": "ok"}
            mock_img.return_value = MagicMock()

            # First execution
            res1 = self.client.post("/analyze", content=body, headers=headers)
            self.assertEqual(res1.status_code, 200)

            # Second identical execution (replay cached result)
            res2 = self.client.post("/analyze", content=body, headers=headers)
            self.assertEqual(res2.status_code, 200)
            self.assertEqual(res1.json(), res2.json())

            # Verify only 1 job record was created
            conn = sqlite3.connect(DB_PATH)
            rows = conn.cursor().execute("SELECT count(*) FROM job_records WHERE event_id = 'evt-dup-test'").fetchone()
            conn.close()
            self.assertEqual(rows[0], 1)

            # Different body with same event_id must yield Conflict 409
            payload_conflict = dict(payload)
            payload_conflict["report"] = {"id": "different-report-id"}
            body_conflict = json.dumps(payload_conflict).encode("utf-8")
            headers_conflict = make_signed_headers("POST", "/analyze", body_conflict, "secret-demo-v1", "demo", "evt-dup-test", "podd-ai-demo-v1")
            res_conflict = self.client.post("/analyze", content=body_conflict, headers=headers_conflict)
            self.assertEqual(res_conflict.status_code, 409)

    # =========================================================================
    # SCENARIO 8: Callback Retry with Saved Payload and Idempotency Key
    # =========================================================================
    async def test_scenario_08_callback_retry_with_saved_payload_and_key(self):
        tenant = self.registry.get_by_schema("demo")
        token_mgr = TenantTokenManager()

        job_key = JobRecord.compute_key("lahis:demo", "podd-ai-demo-v1", "report.submitted", "evt-retry-1")
        job, action = get_or_create_job(
            tenant_key="lahis:demo",
            integration_code="podd-ai-demo-v1",
            operation="report.submitted",
            event_id="evt-retry-1",
            raw_body_bytes=b"raw_bytes",
            request_payload={"test": "data"}
        )

        cluster_payload = {
            "externalClusterId": "cluster-demo-v01-20261016",
            "score": 1.0,
            "riskLevel": "HIGH"
        }
        idempotency_key = "cluster-demo-v01-20261016"

        update_job(job_key, status="CALLBACK_PENDING", callback_payload=cluster_payload)

        # Retrieve job from store and verify stored payload & idempotency key
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        saved_row = conn.cursor().execute("SELECT * FROM job_records WHERE job_key = ?", (job_key,)).fetchone()
        conn.close()

        self.assertEqual(saved_row["status"], "CALLBACK_PENDING")
        self.assertEqual(json.loads(saved_row["callback_payload"]), cluster_payload)

        # Retry callback using exact saved payload and idempotency key
        with patch.object(LahisTenantClient, "submit_cluster", new_callable=AsyncMock) as mock_submit:
            mock_submit.return_value = {"status": "accepted", "id": "cluster-retried-ok"}
            client = LahisTenantClient(tenant, token_mgr)
            resp = await client.submit_cluster(idempotency_key, json.loads(saved_row["callback_payload"]))
            self.assertEqual(resp["id"], "cluster-retried-ok")

            update_job(job_key, status="COMPLETED", callback_response=resp)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        final_row = conn.cursor().execute("SELECT * FROM job_records WHERE job_key = ?", (job_key,)).fetchone()
        conn.close()
        self.assertEqual(final_row["status"], "COMPLETED")

    # =========================================================================
    # SCENARIO 9: Callback Error Responses
    # =========================================================================
    @patch("main.tenant_registry")
    @patch("main.get_genai_client")
    def test_scenario_09_callback_error_responses(self, mock_genai, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        mock_genai.return_value = self._mock_gemini_response()
        tenant = self.registry.get_by_schema("demo")

        payload = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-downstream-err",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-err-test", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-downstream-err", "podd-ai-demo-v1")

        with patch.object(LahisTenantClient, "fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch_imgs, \
             patch.object(LahisTenantClient, "submit_comment", new_callable=AsyncMock) as mock_comment, \
             patch("main.Image.open") as mock_img:

            mock_fetch_imgs.return_value = [b"mock_img"]
            mock_img.return_value = MagicMock()
            # Downstream LAHIS returns 502 with error details
            mock_comment.side_effect = HTTPException(status_code=502, detail="LAHIS comments API error: 500 - Internal Server Error")

            res = self.client.post("/analyze", content=body, headers=headers)
            self.assertEqual(res.status_code, 502)
            self.assertIn("LAHIS comments API error", res.json()["detail"])

            # Verify job is marked as FAILED with error message
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            row = conn.cursor().execute("SELECT status, error_message FROM job_records WHERE event_id = 'evt-downstream-err'").fetchone()
            conn.close()

            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "FAILED")
            self.assertIn("500 - Internal Server Error", row["error_message"])

    # =========================================================================
    # SCENARIO 10: Disabling Tenant A Does Not Affect Tenant B
    # =========================================================================
    @patch("main.tenant_registry")
    @patch("main.get_genai_client")
    def test_scenario_10_disabled_tenant_isolation(self, mock_genai, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        mock_genai.return_value = self._mock_gemini_response()

        tenant_demo = self.registry.get_by_schema("demo")
        tenant_tot = self.registry.get_by_schema("tot")

        # Disable Tenant demo in registry
        disabled_demo = TenantContext(
            tenant_key=tenant_demo.tenant_key,
            deployment_id=tenant_demo.deployment_id,
            schema=tenant_demo.schema,
            code=tenant_demo.code,
            name=tenant_demo.name,
            api_base_url=tenant_demo.api_base_url,
            integration_code=tenant_demo.integration_code,
            oauth_client_id=tenant_demo.oauth_client_id,
            oauth_client_secret=tenant_demo.oauth_client_secret,
            credential_version=tenant_demo.credential_version,
            signing_keys=tenant_demo.signing_keys,
            active_signing_key_id=tenant_demo.active_signing_key_id,
            lang=tenant_demo.lang,
            is_enabled=False  # DISABLED
        )
        self.registry.register(disabled_demo)

        # 1. Tenant demo request must be rejected with 403 Forbidden
        payload_demo = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-dis-demo",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant_demo.api_base_url},
            "report": {"id": "rep-demo-01", "incidentDate": "2026-10-16"}
        }
        body_demo = json.dumps(payload_demo).encode("utf-8")
        headers_demo = make_signed_headers("POST", "/analyze", body_demo, "secret-demo-v1", "demo", "evt-dis-demo", "podd-ai-demo-v1")

        res_demo = self.client.post("/analyze", content=body_demo, headers=headers_demo)
        self.assertEqual(res_demo.status_code, 403)
        self.assertIn("currently disabled", res_demo.json()["detail"])

        # 2. Tenant tot request must succeed normally with 200 OK
        payload_tot = {
            "schemaVersion": "2026-06-02",
            "eventType": "report.submitted",
            "eventId": "evt-en-tot",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "tot", "apiBaseUrl": tenant_tot.api_base_url},
            "report": {"id": "rep-tot-01", "incidentDate": "2026-10-16"}
        }
        body_tot = json.dumps(payload_tot).encode("utf-8")
        headers_tot = make_signed_headers("POST", "/analyze", body_tot, "secret-tot-v1", "tot", "evt-en-tot", "podd-ai-tot-v1")

        with patch.object(LahisTenantClient, "fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
             patch.object(LahisTenantClient, "submit_comment", new_callable=AsyncMock) as mock_comm, \
             patch.object(LahisTenantClient, "submit_cluster", new_callable=AsyncMock) as mock_clust, \
             patch("main.Image.open") as mock_img:

            mock_fetch.return_value = [b"mock"]
            mock_comm.return_value = {"status": "ok"}
            mock_clust.return_value = {"status": "ok"}
            mock_img.return_value = MagicMock()

            res_tot = self.client.post("/analyze", content=body_tot, headers=headers_tot)
            self.assertEqual(res_tot.status_code, 200)

    # =========================================================================
    # SCENARIO 11: Shared Store Failure on GAE (No /tmp Fallback)
    # =========================================================================
    def test_scenario_11_shared_store_failure_blocks_tmp_fallback(self):
        import storage.store as s_store
        orig_require = s_store.REQUIRE_SHARED_STORE
        orig_use = s_store.USE_DATASTORE
        try:
            s_store.REQUIRE_SHARED_STORE = True
            s_store.USE_DATASTORE = True

            with patch("storage.store.get_datastore_client", side_effect=HTTPException(status_code=503, detail="Shared Datastore is required in this deployment environment but could not be initialized.")):
                with self.assertRaises(HTTPException) as ctx:
                    s_store.record_cluster_report(
                        report_id="rep-gae-1",
                        village_id="v-gae-1",
                        village_name="Village GAE",
                        date_str="2026-10-16",
                        raw_payload="{}",
                        detected_disease="Anthrax",
                        tenant_key="lahis:demo"
                    )
                self.assertEqual(ctx.exception.status_code, 503)
                self.assertIn("Shared Datastore is required", ctx.exception.detail)
        finally:
            s_store.REQUIRE_SHARED_STORE = orig_require
            s_store.USE_DATASTORE = orig_use

    # =========================================================================
    # SCENARIO 12: Unsupported Event / Version Rejection
    # =========================================================================
    @patch("main.tenant_registry")
    def test_scenario_12_unsupported_event_and_version_rejection(self, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant = self.registry.get_by_schema("demo")

        # 1. Unsupported schemaVersion
        p1 = {
            "schemaVersion": "1999-01-01",
            "eventType": "report.submitted",
            "eventId": "evt-ver-test",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-1", "incidentDate": "2026-10-16"}
        }
        b1 = json.dumps(p1).encode("utf-8")
        h1 = make_signed_headers("POST", "/analyze", b1, "secret-demo-v1", "demo", "evt-ver-test", "podd-ai-demo-v1")
        r1 = self.client.post("/analyze", content=b1, headers=h1)
        self.assertEqual(r1.status_code, 400)
        self.assertIn("Unsupported schemaVersion", r1.json()["detail"])

        # 2. Unsupported eventType on /analyze
        p2 = {
            "schemaVersion": "2026-06-02",
            "eventType": "unsupported.event.action",
            "eventId": "evt-action-test",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-2", "incidentDate": "2026-10-16"}
        }
        b2 = json.dumps(p2).encode("utf-8")
        h2 = make_signed_headers("POST", "/analyze", b2, "secret-demo-v1", "demo", "evt-action-test", "podd-ai-demo-v1")
        r2 = self.client.post("/analyze", content=b2, headers=h2)
        self.assertEqual(r2.status_code, 400)
        self.assertIn("Unsupported eventType", r2.json()["detail"])

    # =========================================================================
    # SCENARIO 13: Key Rotation
    # =========================================================================
    @patch("main.tenant_registry")
    @patch("main.get_genai_client")
    def test_scenario_13_key_rotation(self, mock_genai, mock_reg):
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        mock_genai.return_value = self._mock_gemini_response()
        tenant = self.registry.get_by_schema("demo")

        with patch.object(LahisTenantClient, "fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
             patch.object(LahisTenantClient, "submit_comment", new_callable=AsyncMock) as mock_comm, \
             patch.object(LahisTenantClient, "submit_cluster", new_callable=AsyncMock) as mock_clust, \
             patch("main.Image.open") as mock_img:

            mock_fetch.return_value = [b"mock"]
            mock_comm.return_value = {"status": "ok"}
            mock_clust.return_value = {"status": "ok"}
            mock_img.return_value = MagicMock()

            payload = {
                "schemaVersion": "2026-06-02",
                "eventType": "report.submitted",
                "eventId": "evt-rot-1",
                "producedAt": datetime.now(timezone.utc).isoformat(),
                "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
                "report": {"id": "rep-rot-1", "incidentDate": "2026-10-16"}
            }
            body = json.dumps(payload).encode("utf-8")

            # 1. Sign with key-demo-1 (secret-demo-v1)
            h1 = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-rot-1", "podd-ai-demo-v1", key_id="key-demo-1")
            r1 = self.client.post("/analyze", content=body, headers=h1)
            self.assertEqual(r1.status_code, 200)

            # 2. Sign with key-demo-2 (secret-demo-v2)
            payload["eventId"] = "evt-rot-2"
            body2 = json.dumps(payload).encode("utf-8")
            h2 = make_signed_headers("POST", "/analyze", body2, "secret-demo-v2", "demo", "evt-rot-2", "podd-ai-demo-v1", key_id="key-demo-2")
            r2 = self.client.post("/analyze", content=body2, headers=h2)
            self.assertEqual(r2.status_code, 200)

            # 3. Revoke key-demo-1 by updating tenant context
            rotated_demo = TenantContext(
                tenant_key=tenant.tenant_key,
                deployment_id=tenant.deployment_id,
                schema=tenant.schema,
                code=tenant.code,
                name=tenant.name,
                api_base_url=tenant.api_base_url,
                integration_code=tenant.integration_code,
                oauth_client_id=tenant.oauth_client_id,
                oauth_client_secret=tenant.oauth_client_secret,
                credential_version=tenant.credential_version,
                signing_keys={"key-demo-2": tenant.signing_keys["key-demo-2"]},  # key-demo-1 REVOKED
                active_signing_key_id="key-demo-2",
                lang=tenant.lang
            )
            self.registry.register(rotated_demo)

            # 4. Attempt to use revoked key-demo-1 -> must return 401
            payload["eventId"] = "evt-rot-3"
            body3 = json.dumps(payload).encode("utf-8")
            h3 = make_signed_headers("POST", "/analyze", body3, "secret-demo-v1", "demo", "evt-rot-3", "podd-ai-demo-v1", key_id="key-demo-1")
            r3 = self.client.post("/analyze", content=body3, headers=h3)
            self.assertEqual(r3.status_code, 401)
            self.assertIn("Unknown signing key ID 'key-demo-1'", r3.json()["detail"])

    def test_default_registry_tot_has_webhook_secret_key_id(self):
        """Verifies that create_default_registry registers INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET for tenant tot."""
        with patch.dict(os.environ, {
            "INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET": "test-env-webhook-secret"
        }):
            reg = create_default_registry()
            tot = reg.get_by_schema("tot")
            self.assertIsNotNone(tot)
            self.assertEqual(tot.active_signing_key_id, "INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET")
            key_obj = tot.get_signing_key("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET")
            self.assertIsNotNone(key_obj)
            self.assertEqual(key_obj.secret, "test-env-webhook-secret")
            # key-tot-v1 should also still be registered as fallback
            self.assertIsNotNone(tot.get_signing_key("key-tot-v1"))

    def test_default_registry_production_credentials(self):
        """Verifies that create_default_registry registers production tenant with client credentials."""
        reg = create_default_registry()
        prod = reg.get_by_schema("production")
        self.assertIsNotNone(prod)
        self.assertEqual(prod.oauth_client_id, "36tX6tXGsQgMALksroK4cFwbZ2fCyCXBPWmz1J4J")
        self.assertEqual(prod.oauth_client_secret, "Czes6besI7YTrvmOWoIaLzNacPNboBCunPR4nCx30yYmTCRiBQLyikiQ7FGsKlNeun6NalSrqhxdDD8D2ZaFUmNtlwmQGhzLaIdwfXUVDSMW5EtlAWhbrH7rmScVE3X8")
        self.assertEqual(prod.api_base_url, "https://prod.api.lahis.ohtk.org")

    @patch("main.tenant_registry")
    def test_webhook_with_integration_podd_fao_ai_webhook_secret_key_id(self, mock_reg):
        """Verifies webhook with X-OHTK-Signing-Key-Id: INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET validates successfully."""
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch("main.get_genai_client", return_value=mock_genai_client), \
             patch("main.Image.open") as mock_img, \
             patch("tenant.client.LahisTenantClient.fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
             patch("tenant.client.LahisTenantClient.submit_comment", new_callable=AsyncMock) as mock_comment:
            mock_fetch.return_value = [b"fake image bytes"]
            mock_comment.return_value = {"id": "comment-123"}

            tenant_tot = self.registry.get_by_schema("tot")
            payload = {
                "schemaVersion": "2026-06-02",
                "eventType": "report.submitted",
                "eventId": "evt-tot-webhook-secret-test",
                "producedAt": datetime.now(timezone.utc).isoformat(),
                "tenant": {"schema": "tot", "apiBaseUrl": tenant_tot.api_base_url},
                "report": {"id": "rep-tot-wh-01", "incidentDate": "2026-10-16"}
            }
            body = json.dumps(payload).encode("utf-8")
            headers = make_signed_headers(
                method="POST",
                path="/analyze",
                raw_body=body,
                secret="secret-tot-v1",
                schema="tot",
                event_id="evt-tot-webhook-secret-test",
                integration_code="podd-ai-tot-v1",
                key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET"
            )

            response = self.client.post("/analyze", content=body, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["is_valid_animal_image"])

    async def test_production_url_and_alias_validation(self):
        """Verifies that production tenant schema and its alias 'prod' validate both prod and production hostnames."""
        reg = create_default_registry()
        prod_tenant = reg.get_by_schema("production")
        self.assertIsNotNone(prod_tenant)
        self.assertEqual(prod_tenant.api_base_url, "https://prod.api.lahis.ohtk.org")

        prod_alias = reg.get_by_schema("prod")
        self.assertIsNotNone(prod_alias)
        self.assertEqual(prod_alias.tenant_key, prod_tenant.tenant_key)

        from tenant.security import validate_webhook_request
        from unittest.mock import MagicMock

        # Test both hostnames are accepted for production
        for host in ("https://prod.api.lahis.ohtk.org", "https://production.api.lahis.ohtk.org"):
            for sch in ("production", "prod"):
                payload = {
                    "schemaVersion": "2026-06-02",
                    "eventType": "report.submitted",
                    "eventId": f"evt-prod-{sch}",
                    "producedAt": datetime.now(timezone.utc).isoformat(),
                    "tenant": {"schema": sch, "apiBaseUrl": host},
                    "report": {"id": "rep-prod-1", "incidentDate": "2026-10-16"}
                }
                body = json.dumps(payload).encode("utf-8")
                headers_dict = make_signed_headers(
                    "POST", "/analyze", body, "test-secret-prod",
                    sch, payload["eventId"], prod_tenant.integration_code,
                    key_id="key-prod-v1"
                )
                # Temporarily add test signing key to prod_tenant
                prod_tenant.signing_keys["key-prod-v1"] = TenantSigningKey(
                    key_id="key-prod-v1", version="1", secret="test-secret-prod"
                )

                from starlette.datastructures import Headers
                mock_request = MagicMock()
                mock_request.method = "POST"
                mock_request.url.path = "/analyze"
                mock_request.url.query = ""
                mock_request.headers = Headers(headers=headers_dict)

                ctx, validated_body = await validate_webhook_request(
                    request=mock_request,
                    raw_body=body,
                    registry=reg,
                    expected_operation="report.submitted"
                )
                self.assertEqual(ctx.tenant_key, "lahis:production")
                self.assertEqual(validated_body["eventId"], f"evt-prod-{sch}")

    @patch("main.tenant_registry")
    def test_schema_version_2026_08_31_supported(self, mock_reg):
        """Verifies that webhook payloads with schemaVersion '2026-08-31' are accepted."""
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant = self.registry.get_by_schema("demo")

        payload = {
            "schemaVersion": "2026-08-31",
            "eventType": "report.submitted",
            "eventId": "evt-schema-2026-08-31",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "demo", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-schema-08-31", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_signed_headers("POST", "/analyze", body, "secret-demo-v1", "demo", "evt-schema-2026-08-31", "podd-ai-demo-v1")

        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch("main.get_genai_client", return_value=mock_genai_client), \
             patch("main.Image.open") as mock_img, \
             patch("tenant.client.LahisTenantClient.fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
             patch("tenant.client.LahisTenantClient.submit_comment", new_callable=AsyncMock) as mock_comment:
            mock_fetch.return_value = [b"fake image bytes"]
            mock_comment.return_value = {"id": "comment-ok"}

            response = self.client.post("/analyze", content=body, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["is_valid_animal_image"])

    @patch("main.tenant_registry")
    def test_ai_evaluation_requested_operation_accepted(self, mock_reg):
        """Verifies that webhook payloads with eventType 'ai.evaluation_requested' are accepted."""
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant = self.registry.get_by_schema("tot")

        payload = {
            "schemaVersion": "2026-08-31",
            "eventType": "ai.evaluation_requested",
            "purpose": "summary",
            "eventId": "evt-eval-req-001",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "tot", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-eval-001", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_signed_headers("POST", "/analyze", body, "secret-tot-v1", "tot", "evt-eval-req-001", "podd-ai-tot-v1", key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET")

        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch("main.get_genai_client", return_value=mock_genai_client), \
             patch("main.Image.open") as mock_img, \
             patch("tenant.client.LahisTenantClient.fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
             patch("tenant.client.LahisTenantClient.submit_comment", new_callable=AsyncMock) as mock_comment:
            mock_fetch.return_value = [b"fake image bytes"]
            mock_comment.return_value = {"id": "comment-ok"}

            response = self.client.post("/analyze", content=body, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["is_valid_animal_image"])

    def test_webhooks_lahis_get_health(self):
        """Verifies GET /webhooks/lahis returns status 200 OK."""
        response = self.client.get("/webhooks/lahis")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("main.tenant_registry")
    def test_webhooks_lahis_post_alias(self, mock_reg):
        """Verifies POST /webhooks/lahis routes to the analyze handler properly."""
        mock_reg.get_by_schema.side_effect = self.registry.get_by_schema
        tenant = self.registry.get_by_schema("tot")

        payload = {
            "schemaVersion": "2026-08-31",
            "eventType": "report.submitted",
            "eventId": "evt-alias-001",
            "producedAt": datetime.now(timezone.utc).isoformat(),
            "tenant": {"schema": "tot", "apiBaseUrl": tenant.api_base_url},
            "report": {"id": "rep-alias-001", "incidentDate": "2026-10-16"}
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_signed_headers("POST", "/webhooks/lahis", body, "secret-tot-v1", "tot", "evt-alias-001", "podd-ai-tot-v1", key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET")

        mock_genai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.usage_metadata = None
        mock_response.text = json.dumps({
            "is_valid_animal_image": True,
            "invalid_reason": None,
            "animal_type": "Cattle",
            "diseases": []
        })
        mock_genai_client.models.generate_content.return_value = mock_response

        with patch("main.get_genai_client", return_value=mock_genai_client), \
             patch("main.Image.open") as mock_img, \
             patch("tenant.client.LahisTenantClient.fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
             patch("tenant.client.LahisTenantClient.submit_comment", new_callable=AsyncMock) as mock_comment:
            mock_fetch.return_value = [b"fake image bytes"]
            mock_comment.return_value = {"id": "comment-ok"}

            response = self.client.post("/webhooks/lahis", content=body, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["is_valid_animal_image"])

    def test_default_registry_shared_signing_secrets(self):
        """Verifies create_default_registry registers signing keys for both tot and production."""
        with patch.dict(os.environ, {
            "INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET": "Qa_v1GGU8ALxojmeMSkzxfxrXnG-hof9xRON9BJ3g4EU2aA-3kMWDCqjHXqjbRaM",
            "LAHIS_SIGNING_SECRET": "Qa_v1GGU8ALxojmeMSkzxfxrXnG-hof9xRON9BJ3g4EU2aA-3kMWDCqjHXqjbRaM"
        }):
            reg = create_default_registry()
            tot = reg.get_by_schema("tot")
            self.assertIsNotNone(tot.get_signing_key("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET"))
            self.assertEqual(tot.get_signing_key("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET").secret, "Qa_v1GGU8ALxojmeMSkzxfxrXnG-hof9xRON9BJ3g4EU2aA-3kMWDCqjHXqjbRaM")

            prod = reg.get_by_schema("production")
            self.assertIsNotNone(prod.get_signing_key("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET"))
            self.assertEqual(prod.get_signing_key("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET").secret, "Qa_v1GGU8ALxojmeMSkzxfxrXnG-hof9xRON9BJ3g4EU2aA-3kMWDCqjHXqjbRaM")
            self.assertIsNotNone(prod.get_signing_key("key-prod-v1"))
            self.assertEqual(prod.integration_code, "cmu-podd-fao-ai-production")

    def test_production_integration_code_cmu_podd_fao_ai_production(self):
        """Verifies X-OHTK-Integration 'cmu-podd-fao-ai-production' is accepted for production."""
        from tenant.models import TenantSigningKey
        reg = create_default_registry()
        prod = reg.get_by_schema("production")
        prod.signing_keys["INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET"] = TenantSigningKey(
            key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET", version="1", secret="test-secret-prod"
        )

        with patch("main.tenant_registry", reg):
            payload = {
                "schemaVersion": "2026-08-31",
                "eventType": "report.submitted",
                "eventId": "evt-cmu-prod-1",
                "producedAt": datetime.now(timezone.utc).isoformat(),
                "tenant": {"schema": "production", "apiBaseUrl": "https://prod.api.lahis.ohtk.org"},
                "report": {"id": "rep-cmu-prod-1", "incidentDate": "2026-10-16"}
            }
            body = json.dumps(payload).encode("utf-8")
            headers = make_signed_headers(
                "POST", "/webhooks/lahis", body, "test-secret-prod",
                "production", payload["eventId"], "cmu-podd-fao-ai-production",
                key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET"
            )

            with patch("tenant.client.LahisTenantClient.fetch_all_report_images_bytes", new_callable=AsyncMock) as mock_fetch, \
                 patch("tenant.client.LahisTenantClient.submit_comment", new_callable=AsyncMock) as mock_comment, \
                 patch("main.Image.open") as mock_img, \
                 patch("main.get_genai_client") as mock_genai:
                mock_fetch.return_value = [b"fake image bytes"]
                mock_comment.return_value = {"id": "comment-ok"}
                mock_genai.return_value = self._mock_gemini_response()

                response = self.client.post("/webhooks/lahis", content=body, headers=headers)
                self.assertEqual(response.status_code, 200)

            # Test mismatch still rejects
            bad_headers = make_signed_headers(
                "POST", "/webhooks/lahis", body, "test-secret-prod",
                "production", payload["eventId"], "completely-invalid-client-code",
                key_id="INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET"
            )
            bad_res = self.client.post("/webhooks/lahis", content=body, headers=bad_headers)
            self.assertEqual(bad_res.status_code, 400)
            self.assertIn("does not match registered client code", bad_res.json()["detail"])


if __name__ == "__main__":
    unittest.main()




