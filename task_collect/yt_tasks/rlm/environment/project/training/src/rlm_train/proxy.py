"""Localhost HTTP proxy: worker sub-LLM calls -> verifiers Client."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


FakeQuery = Callable[[str, "Any"], str | Awaitable[str]]
FakeQueryBatched = Callable[[list[str], "Any"], list[str] | Awaitable[list[str]]]


@dataclass
class ClientHandle:
    client: Any
    model: str
    sampling_args: dict[str, Any] | None = None
    record_call: Any | None = None
    max_concurrent: int = 16
    fake_query: FakeQuery | None = None
    fake_query_batched: FakeQueryBatched | None = None
    state_ref: Any | None = None


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _flatten_prompt(prompt: str | list) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in reversed(prompt):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role != "user":
                continue
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for p in content:
                    t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                    if t:
                        parts.append(str(t))
                return "".join(parts)
            if content is not None:
                return str(content)
    return str(prompt)


def _coerce_messages(prompt: str | list) -> list:
    if isinstance(prompt, str):
        raw: list = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        raw = prompt
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")
    try:
        from verifiers.utils.message_utils import from_raw_message
    except Exception:
        return raw
    out = []
    for m in raw:
        if isinstance(m, dict):
            out.append(from_raw_message(dict(m)))
        else:
            out.append(m)
    return out


class SubLLMProxy:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._host = host
        self._port = port
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._handles: dict[str, ClientHandle] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._runner is not None:
            return
        app = web.Application()
        app.router.add_post("/rollout/{rollout_id}/llm_query", self._handle_single)
        app.router.add_post("/rollout/{rollout_id}/llm_query_batched", self._handle_batched)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        if self._port == 0:
            server = getattr(site, "_server", None)
            socks = getattr(server, "sockets", None) if server else None
            if socks:
                self._port = socks[0].getsockname()[1]
        self._app, self._runner, self._site = app, runner, site
        logger.info("SubLLMProxy listening on %s", self.url)

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._app = self._runner = self._site = None
        self._handles.clear()
        self._semaphores.clear()

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def register(self, rollout_id: str, handle: ClientHandle) -> None:
        self._handles[rollout_id] = handle
        self._semaphores[rollout_id] = asyncio.Semaphore(handle.max_concurrent)

    def unregister(self, rollout_id: str) -> None:
        self._handles.pop(rollout_id, None)
        self._semaphores.pop(rollout_id, None)

    async def _handle_single(self, request: web.Request) -> web.Response:
        rollout_id = request.match_info["rollout_id"]
        handle = self._handles.get(rollout_id)
        if handle is None:
            return web.json_response({"error": f"unknown rollout_id {rollout_id!r}"}, status=404)
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"bad json: {e}"}, status=400)
        prompt = body.get("prompt")
        if prompt is None:
            return web.json_response({"error": "missing 'prompt'"}, status=400)
        model = body.get("model") or handle.model
        try:
            text, meta = await self._completion(handle, prompt, model)
        except Exception as e:  # noqa: BLE001
            logger.exception("sub-llm call failed")
            return web.json_response({"error": str(e)})
        if handle.record_call is not None:
            try:
                handle.record_call({"model": model, "prompt": prompt, "response": text, **meta})
            except Exception:
                logger.exception("record_call failed")
        return web.json_response({"response": text, **meta})

    async def _handle_batched(self, request: web.Request) -> web.Response:
        rollout_id = request.match_info["rollout_id"]
        handle = self._handles.get(rollout_id)
        if handle is None:
            return web.json_response({"error": f"unknown rollout_id {rollout_id!r}"}, status=404)
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"bad json: {e}"}, status=400)
        prompts = body.get("prompts")
        if not isinstance(prompts, list):
            return web.json_response({"error": "missing 'prompts' list"}, status=400)
        model = body.get("model") or handle.model

        if handle.fake_query_batched is not None:
            try:
                result = handle.fake_query_batched(list(prompts), handle.state_ref)
                responses = await _maybe_await(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("fake_query_batched failed")
                return web.json_response({"error": str(e)})
            if responses is not None:
                if not isinstance(responses, list) or len(responses) != len(prompts):
                    return web.json_response({"error": "fake_query_batched returned wrong shape"})
                if handle.record_call is not None:
                    for p, r in zip(prompts, responses, strict=True):
                        try:
                            handle.record_call({"model": model, "prompt": p, "response": r})
                        except Exception:
                            logger.exception("record_call failed")
                return web.json_response(
                    {"responses": [r if isinstance(r, str) else str(r) for r in responses]}
                )

        sem = self._semaphores.get(rollout_id) or asyncio.Semaphore(handle.max_concurrent)

        async def run_one(p: str) -> str:
            async with sem:
                try:
                    text, meta = await self._completion(handle, p, model)
                    if handle.record_call is not None:
                        try:
                            handle.record_call(
                                {"model": model, "prompt": p, "response": text, **meta}
                            )
                        except Exception:
                            logger.exception("record_call failed")
                    return text
                except Exception as e:  # noqa: BLE001
                    logger.exception("sub-llm batched call failed")
                    return f"Error: {e}"

        results = await asyncio.gather(*(run_one(p) for p in prompts))
        return web.json_response({"responses": results})

    async def _completion(
        self,
        handle: ClientHandle,
        prompt: str | list,
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        if handle.fake_query is not None:
            prompt_text = _flatten_prompt(prompt)
            result = handle.fake_query(prompt_text, handle.state_ref)
            content = await _maybe_await(result)
            if content is not None:
                return (content if isinstance(content, str) else str(content)), {}

        messages = _coerce_messages(prompt)
        sampling_args = dict(handle.sampling_args or {})
        # TITO client requires non-None state; hand it an empty trajectory.
        response = await handle.client.get_response(
            prompt=messages,
            model=model,
            tools=None,
            sampling_args=sampling_args,
            state={"trajectory": []},
        )
        try:
            raw_content = response.message.content
        except AttributeError:
            raw_content = None
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts = []
            for p in raw_content:
                text = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                if text:
                    parts.append(text)
            content = "".join(parts)
        else:
            content = ""
        meta: dict[str, Any] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return content, meta
