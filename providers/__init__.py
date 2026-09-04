"""
Provider registry.

`build_provider` maps a provider id plus credentials to a live client, and
`scan_models` is the model-discovery call the settings panel uses right after a
key is pasted.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from providers.base import (
    ChatProvider,
    Message,
    ModelInfo,
    ProviderError,
    StreamEvent,
    ToolCall,
    ToolSpec,
)
from providers.catalog import PROVIDERS, ProviderSpec, get_spec, list_specs
from providers.http_providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAICompatibleProvider,
)
from providers.local_provider import LocalProvider

_BY_PROTOCOL: Dict[str, type] = {
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "local": LocalProvider,
}

__all__ = [
    "ChatProvider", "Message", "ModelInfo", "ProviderError", "StreamEvent",
    "ToolCall", "ToolSpec", "ProviderSpec", "PROVIDERS", "get_spec", "list_specs",
    "build_provider", "scan_models",
]


def build_provider(
    provider_id: str,
    api_key: str = "",
    base_url: str = "",
    engine_factory: Optional[Callable[[], Any]] = None,
    timeout: float = 300.0,
) -> ChatProvider:
    """Instantiates the client for `provider_id`, validating what it needs up front."""
    spec = get_spec(provider_id)
    cls = _BY_PROTOCOL[spec.protocol]

    if spec.requires_key and not api_key:
        raise ProviderError(f"{spec.label} needs an API key.", 401, provider_id)

    resolved_base = (base_url or spec.base_url).rstrip("/")
    if spec.protocol != "local" and not resolved_base:
        raise ProviderError(f"{spec.label} needs a base URL.", 400, provider_id)

    if spec.protocol == "local":
        return LocalProvider(spec, engine_factory=engine_factory, timeout=timeout)
    return cls(spec, api_key=api_key, base_url=resolved_base, timeout=timeout)


async def scan_models(
    provider_id: str,
    api_key: str = "",
    base_url: str = "",
    engine_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """
    Discovers which models the given credentials can actually reach.

    Never raises: a failed scan returns the provider's known-model fallback plus
    the reason, so the UI can still offer a working selection when a provider's
    listing endpoint is missing or restricted.
    """
    spec = get_spec(provider_id)
    try:
        provider = build_provider(provider_id, api_key, base_url, engine_factory)
        models = await provider.list_models()
        await provider.aclose()
        if models:
            return {
                "ok": True,
                "provider": provider_id,
                "source": "live",
                "models": [m.to_dict() for m in models],
            }
        reason = f"{spec.label} returned an empty model list."
    except ProviderError as err:
        reason = str(err)
    except Exception as err:
        reason = f"{spec.label} scan failed: {err}"

    fallback = [ModelInfo(id=mid, owned_by=provider_id).to_dict() for mid in spec.fallback_models]
    return {
        "ok": bool(fallback),
        "provider": provider_id,
        "source": "fallback" if fallback else "none",
        "models": fallback,
        "error": reason,
    }
