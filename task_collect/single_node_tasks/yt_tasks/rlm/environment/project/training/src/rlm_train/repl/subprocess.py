"""Subprocess-based REPL backend: one `python -u worker.py` per rollout."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from rlm_train.repl.base import ExecResult, ReplBackend


class WorkerStartupError(RuntimeError):
    pass


class WorkerProtocolError(RuntimeError):
    pass


class SubprocessReplBackend(ReplBackend):
    _STREAM_LIMIT = 16 * 1024 * 1024

    def __init__(
        self,
        python: str | None = None,
        worker_module: str = "rlm_train.worker",
        startup_timeout: float = 30.0,
        request_timeout: float = 1200.0,
    ):
        self._python = python or sys.executable
        self._worker_module = worker_module
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._req_counter = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        self._stderr_buf: list[str] = []

    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None:
        env = os.environ.copy()
        env["RLM_TRAIN_PROXY_URL"] = proxy_url
        env["RLM_TRAIN_ROLLOUT_ID"] = rollout_id
        env["RLM_TRAIN_DEPTH"] = str(depth)
        env.setdefault("PYTHONUNBUFFERED", "1")

        self._proc = await asyncio.create_subprocess_exec(
            self._python,
            "-u",
            "-m",
            self._worker_module,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=self._STREAM_LIMIT,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            init = await asyncio.wait_for(self._read_line(), timeout=self._startup_timeout)
        except TimeoutError as e:
            await self.stop()
            raise WorkerStartupError(
                f"Worker did not produce init line in {self._startup_timeout}s; "
                f"stderr={self._stderr_text()!r}"
            ) from e
        if not init.get("ok"):
            await self.stop()
            raise WorkerStartupError(f"Worker init failed: {init.get('error')}")

    async def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.returncode is None:
                try:
                    await self._send({"id": "_shutdown", "type": "shutdown"})
                    await asyncio.wait_for(self._read_line(), timeout=2.0)
                except (TimeoutError, BrokenPipeError, ConnectionResetError, WorkerProtocolError):
                    pass
        finally:
            if self._proc.returncode is None:
                try:
                    self._proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()
                try:
                    await self._stderr_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._proc = None
            self._stderr_task = None

    async def load_context(self, payload: Any, index: int | None = None) -> int:
        result = await self._request({"type": "load_context", "payload": payload, "index": index})
        return int(result.get("index", 0))

    async def bootstrap(self, code: str) -> None:
        if not code:
            return
        await self._request({"type": "bootstrap", "code": code})

    async def execute(self, code: str) -> ExecResult:
        result = await self._request({"type": "exec", "code": code})
        return ExecResult(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            final_answer=result.get("final_answer"),
            execution_time=float(result.get("execution_time") or 0.0),
            locals_keys=list(result.get("locals_keys") or []),
        )

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            raise WorkerProtocolError("worker not started")
        async with self._lock:
            self._req_counter += 1
            rid = f"r{self._req_counter}"
            payload = {"id": rid, **payload}
            await self._send(payload)
            try:
                resp = await asyncio.wait_for(self._read_line(), timeout=self._request_timeout)
            except TimeoutError as e:
                await self._kill_worker()
                raise WorkerProtocolError(
                    f"Request {payload.get('type')} exceeded parent watchdog "
                    f"({self._request_timeout:g}s); worker SIGKILLed."
                ) from e
        if resp.get("id") != rid:
            raise WorkerProtocolError(f"id mismatch: expected {rid!r}, got {resp.get('id')!r}")
        if not resp.get("ok"):
            raise WorkerProtocolError(f"worker error: {resp.get('error')}")
        return resp

    async def _kill_worker(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            self._proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except TimeoutError:
            pass

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        line = (json.dumps(payload) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

    async def _read_line(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        line = await self._proc.stdout.readline()
        if not line:
            stderr = self._stderr_text()
            raise WorkerProtocolError(f"worker closed stdout; stderr={stderr!r}")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise WorkerProtocolError(f"bad worker line: {line!r}") from e

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            chunk = await self._proc.stderr.readline()
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            self._stderr_buf.append(text)
            if len(self._stderr_buf) > 200:
                self._stderr_buf = self._stderr_buf[-200:]

    def _stderr_text(self) -> str:
        return "".join(self._stderr_buf)
