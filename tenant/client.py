from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import httpx
from fastapi import HTTPException
from .models import TenantContext
from .auth import TenantTokenManager


class LahisTenantClient:
    """
    HTTP client bound to a specific TenantContext.
    Enforces URL containment, automatic token injection, token refresh on 401,
    and prevents sending bearer tokens across origins or following cross-origin redirects.
    """
    def __init__(self, context: TenantContext, token_manager: TenantTokenManager, client: Optional[httpx.AsyncClient] = None):
        self.context = context
        self.token_manager = token_manager
        self._external_client = client

    def _validate_safe_url(self, target_url: str) -> str:
        """
        Validates that target_url is strictly within the approved tenant origin
        and points to an integration endpoint (/api/integrations/v1/...).
        """
        if target_url.startswith("/"):
            # Relative link
            target_url = f"{self.context.api_base_url}{target_url}"

        parsed_target = urlparse(target_url)
        parsed_approved = urlparse(self.context.api_base_url)

        # Origin check: scheme and host must match approved tenant origin
        approved_hosts = {parsed_approved.netloc.lower()}
        if parsed_approved.netloc.lower() in ("prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org"):
            approved_hosts.update(["prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org"])

        if (parsed_target.scheme.lower() != parsed_approved.scheme.lower() or
                parsed_target.netloc.lower() not in approved_hosts):
            raise HTTPException(
                status_code=400,
                detail=f"Security violation: Target URL origin '{parsed_target.netloc}' does not match approved tenant origin '{parsed_approved.netloc}'"
            )

        # Path check: must be under /api/integrations/v1/ or /o/token/
        path = parsed_target.path
        if not (path.startswith("/api/integrations/v1/") or path.startswith("/o/token/")):
            raise HTTPException(
                status_code=400,
                detail=f"Security violation: Disallowed endpoint path '{path}' for tenant integrations"
            )

        return target_url

    async def _request_with_retry(self, method: str, target_url: str, **kwargs) -> httpx.Response:
        validated_url = self._validate_safe_url(target_url)
        if "prod.api.lahis.ohtk.org" in validated_url:
            validated_url = validated_url.replace("prod.api.lahis.ohtk.org", "production.api.lahis.ohtk.org")
        
        # Follow redirects ONLY within same origin, or disallow redirects
        kwargs.setdefault("follow_redirects", False)
        kwargs.setdefault("timeout", 20.0)

        should_close = False
        client = self._external_client
        if client is None:
            client = httpx.AsyncClient()
            should_close = True

        try:
            token = await self.token_manager.get_token(self.context, client=client)
            headers = kwargs.pop("headers", {}) or {}
            headers["Authorization"] = f"Bearer {token}"

            response = await client.request(method, validated_url, headers=headers, **kwargs)

            # If 401, invalidate cached token and retry ONCE
            if response.status_code == 401:
                print(f"[LAHIS Client] 401 received from {validated_url}. Invalidating token and retrying once...")
                self.token_manager.invalidate(self.context)
                token = await self.token_manager.get_token(self.context, client=client)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.request(method, validated_url, headers=headers, **kwargs)

            # Prevent cross-origin redirect token leakage
            if response.is_redirect and "location" in response.headers:
                loc = response.headers["location"]
                self._validate_safe_url(loc)

            return response
        finally:
            if should_close:
                await client.aclose()

    async def fetch_incident(self, report_id: str) -> Dict[str, Any]:
        """GET /api/integrations/v1/incidents/{report_id}"""
        url = f"{self.context.api_base_url}/api/integrations/v1/incidents/{report_id}"
        resp = await self._request_with_retry("GET", url)
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Incident {report_id} not found in tenant {self.context.tenant_key}")
        if resp.status_code == 403:
            raise HTTPException(status_code=403, detail=f"Access denied reading incident {report_id} in tenant {self.context.tenant_key}")
        resp.raise_for_status()
        return resp.json()

    async def list_report_images(self, report_id: str) -> List[Dict[str, Any]]:
        """GET /api/integrations/v1/reports/{report_id}/images"""
        url = f"{self.context.api_base_url}/api/integrations/v1/reports/{report_id}/images"
        resp = await self._request_with_retry("GET", url)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("images", [])

    async def fetch_image_content(self, content_link_or_url: str) -> bytes:
        """GET /api/integrations/v1/reports/{report_id}/images/{image_id}/content"""
        resp = await self._request_with_retry("GET", content_link_or_url)
        resp.raise_for_status()
        return resp.content

    async def fetch_all_report_images_bytes(self, report_id: str) -> List[bytes]:
        """Helper to list all images and fetch their raw bytes."""
        images = await self.list_report_images(report_id)
        if not images:
            raise HTTPException(status_code=404, detail=f"No animal images found in LAHIS report {report_id}")

        image_bytes_list = []
        for img in images:
            content_link = img.get("links", {}).get("content")
            if content_link:
                try:
                    data = await self.fetch_image_content(content_link)
                    if data:
                        image_bytes_list.append(data)
                except Exception as e:
                    print(f"Warning: Failed to fetch image content {content_link}: {e}")

        if not image_bytes_list:
            raise HTTPException(status_code=404, detail=f"No downloadable image content found in report {report_id}")
        return image_bytes_list

    async def submit_comment(self, report_id: str, idempotency_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/integrations/v1/reports/{report_id}/comments"""
        url = f"{self.context.api_base_url}/api/integrations/v1/reports/{report_id}/comments"
        headers = {
            "Idempotency-Key": idempotency_key,
            "Content-Type": "application/json"
        }
        resp = await self._request_with_retry("POST", url, json=payload, headers=headers)
        if resp.status_code not in (200, 201, 202):
            raise HTTPException(
                status_code=502,
                detail=f"LAHIS comments API error: {resp.status_code} - {resp.text}"
            )
        return resp.json()

    async def submit_cluster(self, idempotency_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /api/integrations/v1/clusters"""
        url = f"{self.context.api_base_url}/api/integrations/v1/clusters"
        headers = {
            "Idempotency-Key": idempotency_key,
            "Content-Type": "application/json"
        }
        resp = await self._request_with_retry("POST", url, json=payload, headers=headers)
        if resp.status_code not in (200, 201, 202):
            raise HTTPException(
                status_code=502,
                detail=f"LAHIS cluster API error: {resp.status_code} - {resp.text}"
            )
        return resp.json()
