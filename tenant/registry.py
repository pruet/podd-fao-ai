import os
import json
from urllib.parse import urlparse
from typing import Dict, List, Optional
from .models import TenantContext, TenantSigningKey


def normalize_api_base_url(url: str, allow_http: bool = False) -> str:
    """
    Normalizes and validates an approved API base URL.
    Must be an exact HTTPS origin (scheme + netloc), with no path, query, fragment, or trailing slash.
    """
    if not url:
        raise ValueError("API base URL cannot be empty")
    
    parsed = urlparse(url)
    if not allow_http and parsed.scheme != "https":
        raise ValueError(f"API base URL must use HTTPS: {url}")
    if allow_http and parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {url}")
    
    if not parsed.netloc:
        raise ValueError(f"Invalid API base URL host: {url}")
    
    # Strip any userinfo, port is allowed in netloc, remove trailing slash and path
    base = f"{parsed.scheme}://{parsed.netloc}"
    return base.rstrip("/")


class TenantRegistry:
    """
    In-memory registry of approved tenants.
    Provides strict lookup and verification without guessing hostnames or cross-tenant fallback.
    """
    def __init__(self, default_deployment_id: str = "lahis"):
        self.default_deployment_id = default_deployment_id
        self._tenants_by_key: Dict[str, TenantContext] = {}
        self._tenants_by_schema: Dict[str, TenantContext] = {}

    def register(self, tenant: TenantContext) -> None:
        """Register a validated TenantContext."""
        self._tenants_by_key[tenant.tenant_key] = tenant
        self._tenants_by_schema[tenant.schema.lower()] = tenant
        if tenant.schema.lower() == "production":
            self._tenants_by_schema["prod"] = tenant
            self._tenants_by_key[f"{tenant.deployment_id}:prod"] = tenant

    def get_by_key(self, tenant_key: str) -> Optional[TenantContext]:
        return self._tenants_by_key.get(tenant_key)

    def get_by_schema(self, schema: str, deployment_id: Optional[str] = None) -> Optional[TenantContext]:
        if not schema:
            return None
        target_dep = deployment_id or self.default_deployment_id
        target_key = f"{target_dep}:{schema.lower()}"
        tenant = self._tenants_by_key.get(target_key)
        if tenant:
            return tenant
        # Fallback to schema lookup if deployment_id matches default
        return self._tenants_by_schema.get(schema.lower())

    def list_all(self) -> List[TenantContext]:
        return list(self._tenants_by_key.values())

    def clear(self) -> None:
        self._tenants_by_key.clear()
        self._tenants_by_schema.clear()


def create_default_registry(config_file_path: Optional[str] = None, allow_http: bool = False) -> TenantRegistry:
    """
    Initializes TenantRegistry from a JSON config file if present,
    otherwise configures the required approved LAHIS mappings:
      - tot: https://demo.api.lahis.ohtk.org
      - demo: https://tot.api.lahis.ohtk.org
      - production: https://production.api.lahis.ohtk.org
    """
    registry = TenantRegistry()

    if config_file_path and os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("tenants", []):
                dep_id = item.get("deployment_id", "lahis")
                schema = item["schema"]
                tenant_key = item.get("tenant_key", f"{dep_id}:{schema}")
                
                # Resolve OAuth credentials
                oauth_info = item.get("oauth", {})
                client_id = oauth_info.get("client_id") or os.getenv(oauth_info.get("client_id_env", ""), "")
                client_secret = oauth_info.get("client_secret") or os.getenv(oauth_info.get("client_secret_env", ""), "")
                cred_ver = oauth_info.get("credential_version", "v1")

                # Resolve Signing keys
                signing_info = item.get("signing", {})
                active_key_id = signing_info.get("active_key_id", "")
                signing_keys = {}
                for k_id, k_val in signing_info.get("keys", {}).items():
                    k_sec = k_val.get("secret") or os.getenv(k_val.get("secret_env", ""), "")
                    signing_keys[k_id] = TenantSigningKey(
                        key_id=k_id,
                        version=str(k_val.get("version", "1")),
                        secret=k_sec,
                        algorithm=k_val.get("algorithm", "HMAC-SHA256")
                    )

                context = TenantContext(
                    tenant_key=tenant_key,
                    deployment_id=dep_id,
                    schema=schema,
                    code=item.get("code", schema),
                    name=item.get("name", f"LAHIS {schema}"),
                    api_base_url=normalize_api_base_url(item["api_base_url"], allow_http=allow_http),
                    integration_code=item.get("integration_code", "cmu-podd-fao-ai-production" if schema in ("production", "prod") else f"podd-ai-{schema}-v1"),
                    oauth_client_id=client_id,
                    oauth_client_secret=client_secret,
                    credential_version=cred_ver,
                    signing_keys=signing_keys,
                    active_signing_key_id=active_key_id,
                    lang=item.get("lang", "lo"),
                    config_revision=item.get("config_revision", "1"),
                    enabled_operations=tuple(item.get("enabled_operations", ["report.submitted", "detect-cluster"])),
                    is_enabled=item.get("is_enabled", True)
                )
                registry.register(context)
        return registry

    # Default Approved LAHIS Tenant Specifications per handoff.md
    default_mappings = [
        {
            "schema": "tot",
            "name": "LAHIS TOT",
            "api_base_url": "https://demo.api.lahis.ohtk.org",
            "integration_code": os.getenv("LAHIS_TOT_INTEGRATION_CODE") or os.getenv("LAHIS_INTEGRATION_CODE") or "podd-fao-ai-staging",
            "client_id": os.getenv("LAHIS_TOT_CLIENT_ID") or (os.getenv("LAHIS_CLIENT_ID") if os.getenv("TENANT_API_URL") in ("https://demo.api.lahis.ohtk.org", "https://tot.api.lahis.ohtk.org") else "") or os.getenv("LAHIS_CLIENT_ID") or "",
            "client_secret": os.getenv("LAHIS_TOT_CLIENT_SECRET") or (os.getenv("LAHIS_CLIENT_SECRET") if os.getenv("TENANT_API_URL") in ("https://demo.api.lahis.ohtk.org", "https://tot.api.lahis.ohtk.org") else "") or os.getenv("LAHIS_CLIENT_SECRET") or "",
            "signing_secret": os.getenv("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET") or os.getenv("LAHIS_TOT_SIGNING_SECRET") or (os.getenv("LAHIS_SIGNING_SECRET") if os.getenv("TENANT_API_URL") in ("https://demo.api.lahis.ohtk.org", "https://tot.api.lahis.ohtk.org") else "") or os.getenv("LAHIS_SIGNING_SECRET") or "",
            "active_key_id": "INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET",
            "additional_key_ids": ["key-tot-v1"]
        },
        {
            "schema": "demo",
            "name": "LAHIS Demo",
            "api_base_url": "https://tot.api.lahis.ohtk.org",
            "integration_code": os.getenv("LAHIS_DEMO_INTEGRATION_CODE") or os.getenv("LAHIS_INTEGRATION_CODE") or "podd-fao-ai-staging",
            "client_id": os.getenv("LAHIS_DEMO_CLIENT_ID") or (os.getenv("LAHIS_CLIENT_ID") if os.getenv("TENANT_API_URL") == "https://tot.api.lahis.ohtk.org" else ""),
            "client_secret": os.getenv("LAHIS_DEMO_CLIENT_SECRET") or (os.getenv("LAHIS_CLIENT_SECRET") if os.getenv("TENANT_API_URL") == "https://tot.api.lahis.ohtk.org" else ""),
            "signing_secret": os.getenv("LAHIS_DEMO_SIGNING_SECRET") or (os.getenv("LAHIS_SIGNING_SECRET") if os.getenv("TENANT_API_URL") == "https://tot.api.lahis.ohtk.org" else ""),
            "active_key_id": "key-demo-v1"
        },
        {
            "schema": "production",
            "name": "LAHIS Production",
            "api_base_url": os.getenv("TENANT_API_URL") if os.getenv("TENANT_API_URL") in ("https://prod.api.lahis.ohtk.org", "https://production.api.lahis.ohtk.org") else "https://prod.api.lahis.ohtk.org",
            "integration_code": os.getenv("LAHIS_PROD_INTEGRATION_CODE") or os.getenv("LAHIS_INTEGRATION_CODE") or "cmu-podd-fao-ai-production",
            "client_id": os.getenv("LAHIS_PROD_CLIENT_ID") or (os.getenv("LAHIS_CLIENT_ID") if os.getenv("TENANT_API_URL") in ("https://prod.api.lahis.ohtk.org", "https://production.api.lahis.ohtk.org") else "") or "36tX6tXGsQgMALksroK4cFwbZ2fCyCXBPWmz1J4J",
            "client_secret": os.getenv("LAHIS_PROD_CLIENT_SECRET") or (os.getenv("LAHIS_CLIENT_SECRET") if os.getenv("TENANT_API_URL") in ("https://prod.api.lahis.ohtk.org", "https://production.api.lahis.ohtk.org") else "") or "Czes6besI7YTrvmOWoIaLzNacPNboBCunPR4nCx30yYmTCRiBQLyikiQ7FGsKlNeun6NalSrqhxdDD8D2ZaFUmNtlwmQGhzLaIdwfXUVDSMW5EtlAWhbrH7rmScVE3X8",
            "signing_secret": os.getenv("LAHIS_PROD_SIGNING_SECRET") or (os.getenv("LAHIS_SIGNING_SECRET") if os.getenv("TENANT_API_URL") in ("https://prod.api.lahis.ohtk.org", "https://production.api.lahis.ohtk.org") else "") or os.getenv("LAHIS_SIGNING_SECRET") or os.getenv("INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET") or "",
            "active_key_id": "INTEGRATION_PODD_FAO_AI_WEBHOOK_SECRET",
            "additional_key_ids": ["key-prod-v1"]
        }
    ]

    for m in default_mappings:
        schema = m["schema"]
        active_key = m["active_key_id"]
        sec = m["signing_secret"]
        signing_keys = {}
        if sec:
            signing_keys[active_key] = TenantSigningKey(
                key_id=active_key,
                version="1",
                secret=sec,
                algorithm="HMAC-SHA256"
            )
            for add_k in m.get("additional_key_ids", []):
                signing_keys[add_k] = TenantSigningKey(
                    key_id=add_k,
                    version="1",
                    secret=sec,
                    algorithm="HMAC-SHA256"
                )

        context = TenantContext(
            tenant_key=f"lahis:{schema}",
            deployment_id="lahis",
            schema=schema,
            code=schema,
            name=m["name"],
            api_base_url=normalize_api_base_url(m["api_base_url"], allow_http=allow_http),
            integration_code=m["integration_code"],
            oauth_client_id=m["client_id"],
            oauth_client_secret=m["client_secret"],
            credential_version="v1",
            signing_keys=signing_keys,
            active_signing_key_id=active_key,
            lang=os.getenv("LAHIS_LANG", "lo"),
            config_revision="2026-10-12",
            enabled_operations=("report.submitted", "detect-cluster", "ai.evaluation_requested"),
            is_enabled=True
        )
        registry.register(context)

    return registry
