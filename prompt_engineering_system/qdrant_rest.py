"""Qdrant Cloud REST client via urllib (Windows-friendly SSL)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def normalize_qdrant_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith("http"):
        url = f"https://{url}"
    if ".cloud.qdrant.io" in url and ":6333" not in url:
        url = f"{url}:6333"
    return url


class QdrantRestClient:
    def __init__(self, url: str, api_key: str, timeout: int = 120):
        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY are required")
        self.base_url = normalize_qdrant_url(url)
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = None
        headers = {"api-key": self.api_key}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qdrant HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qdrant connection failed: {exc}") from exc

    def list_collections(self) -> List[str]:
        payload = self._request("GET", "/collections")
        collections = payload.get("result", {}).get("collections", [])
        return [item["name"] for item in collections]

    def ensure_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> None:
        existing = self.list_collections()
        if name in existing:
            return
        self._request(
            "PUT",
            f"/collections/{name}",
            {
                "vectors": {
                    "size": vector_size,
                    "distance": distance,
                }
            },
        )

    def upsert_points(
        self,
        collection: str,
        points: List[Dict[str, Any]],
    ) -> None:
        self._request(
            "PUT",
            f"/collections/{collection}/points",
            {"points": points},
        )

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        payload = self._request(
            "POST",
            f"/collections/{collection}/points/search",
            {
                "vector": vector,
                "limit": limit,
                "with_payload": True,
            },
        )
        return payload.get("result", [])

    def delete_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{name}")
