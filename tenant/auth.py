import time
import asyncio
from typing import Dict, Tuple, Optional
import httpx
from fastapi import HTTPException
from .models import TenantContext


class TenantTokenManager:
    """
    Manages OAuth tokens strictly scoped to TenantContext.
    Cache key: (tenant_key, integration_code, credential_version)
    Never leaks or shares tokens across tenants.
    """
    def __init__(self):
        # Key: (tenant_key, integration_code, credential_version) -> (access_token, expires_at_timestamp)
        self._cache: Dict[Tuple[str, str, str], Tuple[str, float]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, tenant_key: str) -> asyncio.Lock:
        if tenant_key not in self._locks:
            self._locks[tenant_key] = asyncio.Lock()
        return self._locks[tenant_key]

    def invalidate(self, context: TenantContext) -> None:
        """Evicts cached token for a specific tenant on 401 error."""
        cache_key = (context.tenant_key, context.integration_code, context.credential_version)
        self._cache.pop(cache_key, None)

    async def get_token(self, context: TenantContext, client: Optional[httpx.AsyncClient] = None) -> str:
        cache_key = (context.tenant_key, context.integration_code, context.credential_version)
        now = time.time()

        # Check existing valid token
        if cache_key in self._cache:
            token, expires_at = self._cache[cache_key]
            if expires_at > now + 60:  # 60-second safety margin
                return token

        lock = self._get_lock(context.tenant_key)
        async with lock:
            # Double-check inside lock
            if cache_key in self._cache:
                token, expires_at = self._cache[cache_key]
                if expires_at > now + 60:
                    return token

            if not context.oauth_client_id or not context.oauth_client_secret:
                raise HTTPException(
                    status_code=500,
                    detail=f"OAuth client credentials missing for tenant {context.tenant_key}"
                )

            base_url = context.api_base_url
            if "prod.api.lahis.ohtk.org" in base_url:
                base_url = base_url.replace("prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org")
            token_url = f"{base_url}/o/token/"
            token_data = {
                "grant_type": "client_credentials",
                "client_id": context.oauth_client_id,
                "client_secret": context.oauth_client_secret
            }

            should_close = False
            if client is None:
                client = httpx.AsyncClient()
                should_close = True

            try:
                response = await client.post(
                    token_url,
                    data=token_data,
                    headers={"Accept": "application/json"},
                    timeout=15.0
                )
                if response.status_code in (401, 403):
                    raise HTTPException(
                        status_code=502,
                        detail=f"Failed to authenticate with LAHIS tenant {context.tenant_key}: {response.status_code} {response.text}"
                    )
                response.raise_for_status()
                data = response.json()
                access_token = data.get("access_token")
                expires_in = int(data.get("expires_in", 3600))
                if not access_token:
                    raise HTTPException(
                        status_code=502,
                        detail=f"No access_token returned by LAHIS tenant {context.tenant_key}"
                    )
                self._cache[cache_key] = (access_token, now + expires_in)
                return access_token
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Error requesting OAuth token from LAHIS tenant {context.tenant_key}: {str(e)}"
                )
            finally:
                if should_close:
                    await client.aclose()
