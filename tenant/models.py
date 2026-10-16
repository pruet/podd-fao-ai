from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class TenantSigningKey:
    """Represents a webhook signing key configuration."""
    key_id: str
    version: str
    secret: str
    algorithm: str = "HMAC-SHA256"


@dataclass(frozen=True)
class TenantContext:
    """
    Immutable tenant context representing an authorized, validated tenant.
    Carried through all helper functions and async jobs without relying on global state.
    """
    tenant_key: str                     # Primary identifier: "(deployment_id):(schema)", e.g. "lahis:demo"
    deployment_id: str                  # Deployment partition, e.g. "lahis"
    schema: str                         # Tenant schema slug, e.g. "demo", "tot", "production"
    code: str                           # Tenant code, e.g. "demo"
    name: str                           # Descriptive tenant name
    api_base_url: str                   # Exact approved HTTPS origin without trailing slash
    integration_code: str               # Stable integration code e.g. "podd-ai-demo-v1"
    oauth_client_id: str
    oauth_client_secret: str
    credential_version: str             # e.g. "v1"
    signing_keys: Dict[str, TenantSigningKey] = field(default_factory=dict)
    active_signing_key_id: str = ""
    lang: str = "lo"                    # Default language e.g. "lo", "th", "en"
    config_revision: str = "1"
    enabled_operations: Tuple[str, ...] = ("report.submitted", "detect-cluster", "ai.evaluation_requested")
    is_enabled: bool = True

    def get_signing_key(self, key_id: str) -> Optional[TenantSigningKey]:
        return self.signing_keys.get(key_id)

    def is_operation_enabled(self, operation: str) -> bool:
        return self.is_enabled and (operation in self.enabled_operations)
