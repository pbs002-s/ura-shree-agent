"""
Network-backed providers: the OpenAI chat-completions protocol, the Anthropic
messages protocol, and the Google Gemini protocol.

All three are wrapped into the neutral `Message` / `StreamEvent` shapes from
`providers.base`, including tool calls, so the agent loop never learns which
vendor it is talking to.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from providers.base import (
    ChatProvider,
    Message,
    ModelInfo,
    ProviderError,
    StreamEvent,
    ToolCall,
    ToolSpec,
)

# Substrings that mark a listed model as not usable for chat. Provider listings
# are unfiltered catalogues, so embeddings, image and audio models come back too.
NON_CHAT_HINTS = (
    "embed", "whisper", "tts", "dall-e", "moderation", "audio", "realtime",
    "image", "rerank", "vision-encoder", "guard", "transcribe", "search-",
    "aqa", "veo", "imagen", "learnlm-", "text-bison",
)


def _looks_like_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(hint in lowered for hint in NON_CHAT_HINTS)


async def _request_json(
    client: httpx.AsyncClient, method: str, url: str, provider: str, **kwargs
) -> Dict[str, Any]:
    try:
        response = await client.request(method, url, **kwargs)
    except httpx.HTTPError as err:
        raise ProviderError(f"Could not reach {url}: {err}", provider=provider) from err

    if response.status_code >= 400:
        raise ProviderError(_explain(response, provider), response.status_code, provider)
    try:
        return response.json()
    except ValueError as err:
        raise ProviderError(f"{provider} returned a non-JSON body", response.status_code, provider) from err


def _explain(response: httpx.Response, provider: str) -> str:
    """Turns a provider error body into one actionable sentence."""
    detail = ""
    try:
        body = response.json()
        err = body.get("error", body)
        if isinstance(err, dict):
            detail = err.get("message") or err.get("detail") or ""
        elif isinstance(err, str):
            detail = err
    except ValueError:
        detail = (response.text or "")[:300]

    status = response.status_code
    if status in (401, 403):
        return f"{provider} rejected the API key ({status}). {detail}".strip()
    if status == 404:
        return f"{provider} has no such model or endpoint ({status}). {detail}".strip()
    if status == 429:
        return f"{provider} rate limit or quota exceeded ({status}). {detail}".strip()
    return f"{provider} request failed ({status}). {detail}".strip()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    """Yields parsed JSON payloads from a text/event-stream response."""
    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


class OpenAICompatibleProvider(ChatProvider):
    """
    Speaks POST /chat/completions with `stream: true`.

    This one class covers OpenAI, Groq, OpenRouter, DeepSeek, Mistral, xAI,
    Together, Fireworks, Cerebras, Ollama, LM Studio and any custom endpoint,
    because they all implement the same route.
    """

    protocol = "openai"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter" in self.base_url:
            # OpenRouter attributes requests to an app; harmless elsewhere.
            headers["HTTP-Referer"] = "https://github.com/pbs002-s"
            headers["X-Title"] = "URA-Shree"
        return headers

    async def list_models(self) -> List[ModelInfo]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _request_json(
                client, "GET", f"{self.base_url}{self.spec.models_path}",
                provider=self.spec.label, headers=self._headers(),
            )

        raw = data.get("data") if isinstance(data, dict) else data
        if not isinstance(raw, list):
            raw = []

        models: List[ModelInfo] = []
        for item in raw:
            if isinstance(item, str):
                model_id, meta = item, {}
            elif isinstance(item, dict):
                model_id = item.get("id") or item.get("name") or ""
                meta = item
            else:
                continue
            if not model_id or not _looks_like_chat_model(model_id):
                continue
            models.append(
                ModelInfo(
                    id=model_id,
                    label=meta.get("name") or model_id,
                    context_window=int(
                        meta.get("context_length")
                        or meta.get("context_window")
                        or (meta.get("top_provider") or {}).get("context_length")
                        or 0
                    ),
                    owned_by=str(meta.get("owned_by") or meta.get("organization") or ""),
                )
            )

        models.sort(key=lambda m: m.id)
        return models

    def _payload(
        self,
        model: str,
        messages: List[Message],
        system: str,
        tools: Optional[List[ToolSpec]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        wire: List[Dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "tool":
                wire.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                wire.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                wire.append({"role": msg.role, "content": msg.content})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": wire,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t.to_dict()} for t in tools
            ]
            payload["tool_choice"] = "auto"
        return payload

    async def stream(
        self,
        model: str,
        messages: List[Message],
        system: str = "",
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamEvent, None]:
        payload = self._payload(model, messages, system, tools, temperature, max_tokens)
        # Tool call arguments arrive as string fragments keyed by index.
        pending: Dict[int, Dict[str, str]] = {}
        stop_reason = "end_turn"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                headers=self._headers(), json=payload,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise ProviderError(
                        _explain(response, self.spec.label), response.status_code, self.spec.id
                    )

                in_think = False
                async for chunk in _iter_sse(response):
                    choices = chunk.get("choices") or []
                    if usage := chunk.get("usage"):
                        yield StreamEvent(type="usage", usage=usage)
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    if reasoning := (delta.get("reasoning_content") or delta.get("reasoning")):
                        yield StreamEvent(type="thinking", text=reasoning)

                    if content := delta.get("content"):
                        if "<think>" in content or "</think>" in content or in_think:
                            curr = content
                            while curr:
                                if in_think:
                                    if "</think>" in curr:
                                        th, _, rest = curr.partition("</think>")
                                        in_think = False
                                        if th:
                                            yield StreamEvent(type="thinking", text=th)
                                        curr = rest
                                    else:
                                        yield StreamEvent(type="thinking", text=curr)
                                        curr = ""
                                else:
                                    if "<think>" in curr:
                                        txt, _, rest = curr.partition("<think>")
                                        in_think = True
                                        if txt:
                                            yield StreamEvent(type="text", text=txt)
                                        curr = rest
                                    else:
                                        yield StreamEvent(type="text", text=curr)
                                        curr = ""
                        else:
                            yield StreamEvent(type="text", text=content)

                    for call in delta.get("tool_calls") or []:
                        index = call.get("index", 0)
                        slot = pending.setdefault(index, {"id": "", "name": "", "args": ""})
                        if call.get("id"):
                            slot["id"] = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            slot["name"] = function["name"]
                        if function.get("arguments"):
                            slot["args"] += function["arguments"]

                    if finish := choice.get("finish_reason"):
                        stop_reason = finish

        for index in sorted(pending):
            slot = pending[index]
            if not slot["name"]:
                continue
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=_safe_json(slot["args"]),
                ),
            )

        yield StreamEvent(type="done", stop_reason=stop_reason)


class AnthropicProvider(ChatProvider):
    """Speaks POST /v1/messages with `stream: true`."""

    protocol = "anthropic"
    API_VERSION = "2023-06-01"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }

    async def list_models(self) -> List[ModelInfo]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _request_json(
                client, "GET", f"{self.base_url}/models?limit=100",
                provider=self.spec.label, headers=self._headers(),
            )
        return [
            ModelInfo(
                id=item["id"],
                label=item.get("display_name") or item["id"],
                owned_by="anthropic",
            )
            for item in data.get("data", [])
            if item.get("id")
        ]

    async def stream(
        self,
        model: str,
        messages: List[Message],
        system: str = "",
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamEvent, None]:
        wire: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                # Tool results are user-role blocks in this protocol.
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content,
                }
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            elif msg.role == "assistant" and msg.tool_calls:
                blocks: List[Dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                blocks.extend(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    for tc in msg.tool_calls
                )
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": msg.role, "content": msg.content})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": wire,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        # tool_use blocks stream their input as partial JSON keyed by block index.
        blocks_in_flight: Dict[int, Dict[str, Any]] = {}
        stop_reason = "end_turn"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/messages", headers=self._headers(), json=payload
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise ProviderError(
                        _explain(response, self.spec.label), response.status_code, self.spec.id
                    )

                async for event in _iter_sse(response):
                    kind = event.get("type")

                    if kind == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            blocks_in_flight[event.get("index", 0)] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "args": "",
                            }

                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            yield StreamEvent(type="text", text=delta.get("text", ""))
                        elif dtype == "thinking_delta":
                            yield StreamEvent(type="thinking", text=delta.get("thinking", ""))
                        elif dtype == "input_json_delta":
                            slot = blocks_in_flight.get(event.get("index", 0))
                            if slot is not None:
                                slot["args"] += delta.get("partial_json", "")

                    elif kind == "content_block_stop":
                        slot = blocks_in_flight.pop(event.get("index", -1), None)
                        if slot and slot["name"]:
                            yield StreamEvent(
                                type="tool_call",
                                tool_call=ToolCall(
                                    id=slot["id"],
                                    name=slot["name"],
                                    arguments=_safe_json(slot["args"]),
                                ),
                            )

                    elif kind == "message_delta":
                        stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
                        if usage := event.get("usage"):
                            yield StreamEvent(type="usage", usage=usage)

                    elif kind == "error":
                        message = (event.get("error") or {}).get("message", "stream error")
                        raise ProviderError(message, provider=self.spec.id)

        yield StreamEvent(type="done", stop_reason=stop_reason)


class GoogleProvider(ChatProvider):
    """Speaks POST /v1beta/models/{model}:streamGenerateContent?alt=sse."""

    protocol = "google"

    async def list_models(self) -> List[ModelInfo]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _request_json(
                client, "GET", f"{self.base_url}/models",
                provider=self.spec.label, params={"key": self.api_key, "pageSize": 200},
            )

        models: List[ModelInfo] = []
        for item in data.get("models", []):
            name = (item.get("name") or "").removeprefix("models/")
            if not name or not _looks_like_chat_model(name):
                continue
            if "generateContent" not in (item.get("supportedGenerationMethods") or []):
                continue
            models.append(
                ModelInfo(
                    id=name,
                    label=item.get("displayName") or name,
                    context_window=int(item.get("inputTokenLimit") or 0),
                    owned_by="google",
                )
            )
        return models

    async def stream(
        self,
        model: str,
        messages: List[Message],
        system: str = "",
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamEvent, None]:
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.name or "tool",
                            "response": {"result": msg.content},
                        }
                    }],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                parts: List[Dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    fc_part: Dict[str, Any] = {
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    }
                    sig = getattr(tc, "thought_signature", None) or "skip_thought_signature_validator"
                    fc_part["thoughtSignature"] = sig
                    parts.append(fc_part)
                contents.append({"role": "model", "parts": parts})
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in tools
                ]
            }]

        url = f"{self.base_url}/models/{model}:streamGenerateContent"
        call_index = 0
        stop_reason = "end_turn"
        latest_thought_signature: Optional[str] = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, params={"key": self.api_key, "alt": "sse"}, json=payload
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise ProviderError(
                        _explain(response, self.spec.label), response.status_code, self.spec.id
                    )

                async for chunk in _iter_sse(response):
                    for candidate in chunk.get("candidates", []):
                        for part in (candidate.get("content") or {}).get("parts", []):
                            if text := part.get("text"):
                                yield StreamEvent(type="text", text=text)
                            sig = (
                                part.get("thoughtSignature")
                                or part.get("thought_signature")
                                or (part.get("functionCall") or {}).get("thoughtSignature")
                                or (part.get("functionCall") or {}).get("thought_signature")
                            )
                            if sig:
                                latest_thought_signature = sig
                            if call := part.get("functionCall"):
                                call_index += 1
                                call_sig = (
                                    part.get("thoughtSignature")
                                    or part.get("thought_signature")
                                    or call.get("thoughtSignature")
                                    or call.get("thought_signature")
                                    or latest_thought_signature
                                )
                                yield StreamEvent(
                                    type="tool_call",
                                    tool_call=ToolCall(
                                        id=f"call_{call_index}",
                                        name=call.get("name", ""),
                                        arguments=call.get("args") or {},
                                        thought_signature=call_sig,
                                    ),
                                )
                        if reason := candidate.get("finishReason"):
                            stop_reason = reason
                    if usage := chunk.get("usageMetadata"):
                        yield StreamEvent(type="usage", usage=usage)

        yield StreamEvent(type="done", stop_reason=stop_reason)


def _safe_json(raw: str) -> Dict[str, Any]:
    """Parses accumulated tool arguments, tolerating an empty or broken stream."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_unparsed_arguments": raw}
