"""
Known AI providers.

Every entry describes how to reach a provider: which wire protocol it speaks,
where its base URL is, how the key is presented, and where its model list
lives. Adding a provider means adding a row here - nothing else in the codebase
needs to change.

The three wire protocols cover essentially the whole market:
  openai    - /v1/chat/completions, used by OpenAI and by every service that
              advertises an "OpenAI-compatible endpoint" (most of them)
  anthropic - /v1/messages
  google    - /v1beta/models/{model}:streamGenerateContent
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class ProviderSpec:
    id: str
    label: str
    protocol: str                  # "openai" | "anthropic" | "google" | "local"
    base_url: str
    docs_url: str = ""
    key_url: str = ""
    key_prefix: str = ""           # shown as a hint, never enforced
    requires_key: bool = True
    models_path: str = "/models"
    supports_tools: bool = True
    # Models the provider is known to serve, used when its listing endpoint is
    # unavailable or returns everything including non-chat models.
    fallback_models: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PROVIDERS: Dict[str, ProviderSpec] = {
    "local": ProviderSpec(
        id="local",
        label="URA-Shree (local)",
        protocol="local",
        base_url="",
        requires_key=False,
        supports_tools=True,
        notes="The model trained in this repository. Runs on your own GPU or CPU, no network.",
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic",
        protocol="anthropic",
        base_url="https://api.anthropic.com/v1",
        docs_url="https://docs.anthropic.com",
        key_url="https://console.anthropic.com/settings/keys",
        key_prefix="sk-ant-",
        fallback_models=[
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        protocol="openai",
        base_url="https://api.openai.com/v1",
        docs_url="https://platform.openai.com/docs",
        key_url="https://platform.openai.com/api-keys",
        key_prefix="sk-",
        fallback_models=["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o3-mini"],
    ),
    "google": ProviderSpec(
        id="google",
        label="Google Gemini",
        protocol="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        docs_url="https://ai.google.dev/docs",
        key_url="https://aistudio.google.com/apikey",
        key_prefix="AIza",
        fallback_models=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    ),
    "groq": ProviderSpec(
        id="groq",
        label="Groq",
        protocol="openai",
        base_url="https://api.groq.com/openai/v1",
        docs_url="https://console.groq.com/docs",
        key_url="https://console.groq.com/keys",
        key_prefix="gsk_",
        notes="Very high tokens per second on open-weight models.",
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        protocol="openai",
        base_url="https://openrouter.ai/api/v1",
        docs_url="https://openrouter.ai/docs",
        key_url="https://openrouter.ai/keys",
        key_prefix="sk-or-",
        notes="One key, several hundred models from many vendors.",
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        protocol="openai",
        base_url="https://api.deepseek.com/v1",
        key_url="https://platform.deepseek.com/api_keys",
        key_prefix="sk-",
        fallback_models=["deepseek-chat", "deepseek-reasoner"],
    ),
    "mistral": ProviderSpec(
        id="mistral",
        label="Mistral",
        protocol="openai",
        base_url="https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys",
    ),
    "xai": ProviderSpec(
        id="xai",
        label="xAI Grok",
        protocol="openai",
        base_url="https://api.x.ai/v1",
        key_url="https://console.x.ai",
        key_prefix="xai-",
    ),
    "together": ProviderSpec(
        id="together",
        label="Together AI",
        protocol="openai",
        base_url="https://api.together.xyz/v1",
        key_url="https://api.together.ai/settings/api-keys",
    ),
    "fireworks": ProviderSpec(
        id="fireworks",
        label="Fireworks AI",
        protocol="openai",
        base_url="https://api.fireworks.ai/inference/v1",
        key_url="https://fireworks.ai/account/api-keys",
    ),
    "cerebras": ProviderSpec(
        id="cerebras",
        label="Cerebras",
        protocol="openai",
        base_url="https://api.cerebras.ai/v1",
        key_url="https://cloud.cerebras.ai",
    ),
    "ollama": ProviderSpec(
        id="ollama",
        label="Ollama (local server)",
        protocol="openai",
        base_url="http://localhost:11434/v1",
        requires_key=False,
        docs_url="https://ollama.com",
        notes="Models you have pulled with ollama. No key needed.",
    ),
    "lmstudio": ProviderSpec(
        id="lmstudio",
        label="LM Studio (local server)",
        protocol="openai",
        base_url="http://localhost:1234/v1",
        requires_key=False,
        docs_url="https://lmstudio.ai",
        notes="Start the LM Studio local server first.",
    ),
    "custom": ProviderSpec(
        id="custom",
        label="Custom OpenAI-compatible",
        protocol="openai",
        base_url="",
        requires_key=False,
        notes="Any endpoint that implements /v1/chat/completions. Set the base URL yourself.",
    ),
}


def get_spec(provider_id: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider_id)
    if spec is None:
        raise KeyError(f"Unknown provider '{provider_id}'. Known: {', '.join(PROVIDERS)}")
    return spec


def list_specs() -> List[Dict[str, Any]]:
    return [spec.to_dict() for spec in PROVIDERS.values()]
