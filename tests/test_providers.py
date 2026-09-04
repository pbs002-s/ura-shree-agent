"""
Provider layer: catalogue, request shaping and streamed response parsing.

The network is stubbed with a transport, so these run offline while still
exercising the real httpx path, the real SSE parser and the real tool-call
assembly.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers import build_provider, get_spec, list_specs, scan_models
from providers.base import Message, ProviderError, ToolCall, ToolSpec
from providers.http_providers import AnthropicProvider, OpenAICompatibleProvider

TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
)


def sse(*chunks: dict) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


async def drain(provider, **kwargs):
    return [event async for event in provider.stream(**kwargs)]


def install(monkeypatch, handler):
    """Routes every httpx request in the process through `handler`."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched(self, *args, **kw):
        kw["transport"] = transport
        original(self, *args, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)


def test_catalogue_covers_the_major_providers():
    ids = {spec["id"] for spec in list_specs()}
    for expected in ("local", "openai", "anthropic", "google", "groq", "openrouter", "ollama", "custom"):
        assert expected in ids
    assert get_spec("anthropic").protocol == "anthropic"
    assert get_spec("groq").protocol == "openai"


def test_a_provider_needing_a_key_refuses_without_one():
    with pytest.raises(ProviderError) as err:
        build_provider("openai", api_key="")
    assert "API key" in str(err.value)
    # Local providers and local servers must not demand one.
    build_provider("ollama")
    build_provider("local")


def test_openai_stream_assembles_text_and_tool_calls(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": "Reading "}}]},
            {"choices": [{"delta": {"content": "the file."}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 11, "completion_tokens": 4}},
        ), headers={"content-type": "text/event-stream"})

    install(monkeypatch, handler)
    provider = OpenAICompatibleProvider(get_spec("openai"), api_key="sk-test")

    import asyncio
    events = asyncio.run(drain(
        provider, model="gpt-4.1", messages=[Message(role="user", content="hi")],
        system="be brief", tools=[TOOL],
    ))

    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert captured["body"]["tools"][0]["function"]["name"] == "read_file"

    assert "".join(e.text for e in events if e.type == "text") == "Reading the file."
    call = next(e.tool_call for e in events if e.type == "tool_call")
    # Arguments arrive split across frames and must be reassembled before parsing.
    assert call.name == "read_file" and call.arguments == {"path": "a.py"}
    assert any(e.type == "usage" for e in events)
    assert events[-1].type == "done" and events[-1].stop_reason == "tool_calls"


def test_anthropic_stream_and_tool_result_shape(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, content=sse(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Checking."}},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file"}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta", "partial_json": '{"path": "b.py"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
             "usage": {"output_tokens": 9}},
        ), headers={"content-type": "text/event-stream"})

    install(monkeypatch, handler)
    provider = AnthropicProvider(get_spec("anthropic"), api_key="sk-ant-test")

    import asyncio
    events = asyncio.run(drain(
        provider,
        model="claude-sonnet-4-6",
        messages=[
            Message(role="user", content="look at b.py"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="toolu_0", name="read_file", arguments={"path": "b.py"})]),
            Message(role="tool", content="file body", tool_call_id="toolu_0", name="read_file"),
        ],
        system="be brief",
        tools=[TOOL],
    ))

    assert captured["key"] == "sk-ant-test"
    # The system prompt is a top-level field in this protocol, not a message.
    assert captured["body"]["system"] == "be brief"
    assert captured["body"]["tools"][0]["input_schema"]["required"] == ["path"]
    # A tool result must be sent as a user-role tool_result block.
    last = captured["body"]["messages"][-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"

    assert "".join(e.text for e in events if e.type == "text") == "Checking."
    call = next(e.tool_call for e in events if e.type == "tool_call")
    assert call.id == "toolu_1" and call.arguments == {"path": "b.py"}
    assert events[-1].stop_reason == "tool_use"


def test_error_bodies_become_readable_messages(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    install(monkeypatch, handler)
    provider = OpenAICompatibleProvider(get_spec("openai"), api_key="sk-bad")

    import asyncio
    with pytest.raises(ProviderError) as err:
        asyncio.run(drain(provider, model="gpt-4.1", messages=[Message(role="user", content="hi")]))

    assert err.value.status == 401
    assert "rejected the API key" in str(err.value)
    assert "Incorrect API key provided" in str(err.value)


def test_model_scan_filters_out_non_chat_models(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"id": "gpt-4.1", "owned_by": "openai"},
            {"id": "text-embedding-3-small"},
            {"id": "whisper-1"},
            {"id": "dall-e-3"},
        ]})

    install(monkeypatch, handler)

    import asyncio
    result = asyncio.run(scan_models("openai", api_key="sk-test"))
    assert result["ok"] and result["source"] == "live"
    assert [m["id"] for m in result["models"]] == ["gpt-4.1"]


def test_a_failed_scan_falls_back_instead_of_raising(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream exploded"})

    install(monkeypatch, handler)

    import asyncio
    result = asyncio.run(scan_models("anthropic", api_key="sk-ant-test"))
    # The UI still needs something selectable, and the reason must be reported.
    assert result["source"] == "fallback"
    assert result["models"] and result["error"]
