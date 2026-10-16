import os
import hmac
import hashlib
import json
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional
from fastapi import Request, HTTPException
from .models import TenantContext
from .registry import TenantRegistry, normalize_api_base_url


MAX_BODY_SIZE_BYTES = 1024 * 1024  # 1 MB
TIMESTAMP_TOLERANCE_SECONDS = 300   # ±5 minutes


def verify_hmac_signature(
    method: str,
    path_with_query: str,
    timestamp: str,
    raw_body: bytes,
    secret: str,
    provided_signature: str
) -> bool:
    """
    Computes HMAC-SHA256 over:
    METHOD + "\n" + path_with_query + "\n" + timestamp_header + "\n" + raw_body_bytes
    and compares against provided_signature in constant time.
    """
    if not secret or not provided_signature:
        return False
    message = f"{method.upper()}\n{path_with_query}\n{timestamp}\n".encode("utf-8") + raw_body
    computed_sig = hmac.new(
        secret.encode("utf-8"),
        message,
        hashlib.sha256
    ).hexdigest().lower()
    return hmac.compare_digest(computed_sig, provided_signature.lower())


def validate_timestamp_freshness(timestamp_str: str, now: Optional[datetime] = None) -> datetime:
    """
    Parses ISO-8601 timestamp with timezone and verifies freshness within ±5 minutes.
    """
    if not timestamp_str:
        raise HTTPException(status_code=401, detail="Missing X-OHTK-Timestamp header")

    try:
        # Support both 'Z' and offset notations
        ts_clean = timestamp_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid ISO-8601 X-OHTK-Timestamp: {str(e)}")

    if dt.tzinfo is None:
        raise HTTPException(status_code=401, detail="X-OHTK-Timestamp must include timezone")

    current_time = now or datetime.now(timezone.utc)
    delta_seconds = abs((current_time - dt).total_seconds())

    if delta_seconds > TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=401,
            detail=f"X-OHTK-Timestamp drift exceeds tolerance window: {delta_seconds:.1f}s > {TIMESTAMP_TOLERANCE_SECONDS}s"
        )

    return dt


SUPPORTED_SCHEMA_VERSIONS = {"2026-06-02", "2026-08-31"}


async def validate_webhook_request(
    request: Request,
    raw_body: bytes,
    registry: TenantRegistry,
    expected_operation: Any = "report.submitted",
    now: Optional[datetime] = None
) -> Tuple[TenantContext, Dict[str, Any]]:
    """
    Executes the strict 7-step validation pipeline defined in handoff.md.
    Returns (TenantContext, parsed_body_dict).
    """
    # 1. Limit body size and check required fields
    if len(raw_body) > MAX_BODY_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"Request body exceeds {MAX_BODY_SIZE_BYTES} bytes limit")

    try:
        body = json.loads(raw_body)
        if not isinstance(body, dict):
            raise ValueError("Root body must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

    # Check required top-level fields
    event_type = body.get("eventType")
    event_id = body.get("eventId")
    schema_version = body.get("schemaVersion")
    tenant_obj = body.get("tenant") or {}
    tenant_schema = tenant_obj.get("schema")

    if not all([event_type, event_id, tenant_schema]):
        raise HTTPException(status_code=400, detail="Missing required webhook fields: eventType, eventId, tenant.schema")

    if schema_version and schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported schemaVersion '{schema_version}'. Supported versions: {sorted(list(SUPPORTED_SCHEMA_VERSIONS))}"
        )

    if expected_operation:
        if isinstance(expected_operation, (list, tuple, set)):
            if event_type not in expected_operation:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported eventType '{event_type}'. This endpoint requires one of {list(expected_operation)}"
                )
        elif event_type != expected_operation:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported eventType '{event_type}'. This endpoint requires '{expected_operation}'"
            )

    # 2. Candidate Tenant Lookup (strictly by schema, no guessing)
    context = registry.get_by_schema(tenant_schema)
    if not context:
        raise HTTPException(status_code=404, detail=f"Unknown tenant schema '{tenant_schema}'")

    if not context.is_enabled:
        raise HTTPException(status_code=403, detail=f"Tenant '{tenant_schema}' is currently disabled")

    # 3. Extract and check headers
    h_tenant = request.headers.get("x-ohtk-tenant")
    h_event_id = request.headers.get("x-ohtk-event-id")
    h_integration = request.headers.get("x-ohtk-integration")
    h_timestamp = request.headers.get("x-ohtk-timestamp")
    h_signature = request.headers.get("x-ohtk-signature")
    h_alg = request.headers.get("x-ohtk-signature-alg", "HMAC-SHA256")
    h_key_id = request.headers.get("x-ohtk-signing-key-id")
    h_sec_version = request.headers.get("x-ohtk-signing-secret-version")

    if not h_signature or not h_timestamp:
        raise HTTPException(status_code=401, detail="Missing X-OHTK-Signature or X-OHTK-Timestamp headers")

    if h_alg.upper() != "HMAC-SHA256":
        raise HTTPException(status_code=401, detail=f"Unsupported signature algorithm '{h_alg}'. Must be HMAC-SHA256")

    # Resolve signing key for tenant
    signing_key = None
    if h_key_id:
        signing_key = context.get_signing_key(h_key_id)
        if not signing_key:
            raise HTTPException(
                status_code=401,
                detail=f"Unknown signing key ID '{h_key_id}' for tenant '{tenant_schema}'"
            )
        if h_sec_version and str(signing_key.version) != str(h_sec_version):
            raise HTTPException(
                status_code=401,
                detail=f"Signing key version mismatch: expected {signing_key.version}, got {h_sec_version}"
            )
    else:
        # Fall back to active signing key if none specified in header
        signing_key = context.get_signing_key(context.active_signing_key_id)

    if not signing_key or not signing_key.secret:
        raise HTTPException(
            status_code=401,
            detail=f"No valid signing secret configured for tenant '{tenant_schema}'"
        )

    # 4. Check constant-time HMAC signature over exact path
    path_with_query = request.url.path
    if request.url.query:
        path_with_query += f"?{request.url.query}"

    sig_ok = verify_hmac_signature(
        method=request.method,
        path_with_query=path_with_query,
        timestamp=h_timestamp,
        raw_body=raw_body,
        secret=signing_key.secret,
        provided_signature=h_signature
    )
    if not sig_ok:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # 5. Timestamp freshness check
    validate_timestamp_freshness(h_timestamp, now=now)

    # 6. Header cross-checks with signed body
    if h_tenant and h_tenant.lower() != tenant_schema.lower():
        raise HTTPException(
            status_code=400,
            detail=f"X-OHTK-Tenant header '{h_tenant}' does not match body tenant.schema '{tenant_schema}'"
        )

    if h_event_id and h_event_id != event_id:
        raise HTTPException(
            status_code=400,
            detail=f"X-OHTK-Event-ID header '{h_event_id}' does not match body eventId '{event_id}'"
        )

    if h_integration:
        allowed_integration_codes = {
            context.integration_code,
            "cmu-podd-fao-ai-production",
            "podd-fao-ai-staging",
            "cmu-podd-fao-ai-staging",
            "podd-ai-tot-v1",
            "podd-ai-demo-v1",
            "podd-ai-prod-v1"
        }
        for env_var in ("LAHIS_INTEGRATION_CODE", "LAHIS_PROD_INTEGRATION_CODE", "LAHIS_TOT_INTEGRATION_CODE", "LAHIS_DEMO_INTEGRATION_CODE"):
            val = os.getenv(env_var)
            if val:
                allowed_integration_codes.add(val)

        if h_integration not in allowed_integration_codes:
            raise HTTPException(
                status_code=400,
                detail=f"X-OHTK-Integration header '{h_integration}' does not match registered client code '{context.integration_code}'"
            )

    # 7. Check signed apiBaseUrl and operation enablement
    body_api_url = tenant_obj.get("apiBaseUrl")
    if not body_api_url:
        raise HTTPException(status_code=400, detail="Missing tenant.apiBaseUrl in signed payload")

    norm_body_url = normalize_api_base_url(body_api_url, allow_http=True)
    url_matches = (norm_body_url.lower() == context.api_base_url.lower())
    if not url_matches and context.schema in ("production", "prod"):
        allowed_prod_urls = {"https://prod.api.lahis.ohtk.org", "https://production.api.lahis.ohtk.org"}
        if norm_body_url.lower() in allowed_prod_urls and context.api_base_url.lower() in allowed_prod_urls:
            url_matches = True

    if not url_matches:
        raise HTTPException(
            status_code=400,
            detail=f"Signed tenant.apiBaseUrl '{norm_body_url}' does not match approved registry URL '{context.api_base_url}'"
        )

    if not context.is_operation_enabled(event_type):
        raise HTTPException(
            status_code=403,
            detail=f"Operation '{event_type}' is disabled for tenant '{tenant_schema}'"
        )

    return context, body
