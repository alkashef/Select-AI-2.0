"""Run a persistent asyncio loop in a background thread."""

from __future__ import annotations

import asyncio
from threading import Event, Thread
from typing import Any


class EventLoopThread(Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._loop = asyncio.new_event_loop()
        self._stop_event = Event()

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_coroutine(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()

        def _cancel_pending() -> None:
            tasks = [task for task in asyncio.all_tasks(loop=self._loop) if not task.done()]
            for task in tasks:
                task.cancel()
            self._loop.stop()

        self._loop.call_soon_threadsafe(_cancel_pending)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop
