import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
import pytest

from sglang.multimodal_gen.runtime.entrypoints.http_server import lifespan


def test_lifespan_fails_startup_when_broker_bind_fails(monkeypatch):
    async def run_test():
        bind_error = RuntimeError("broker bind failed")

        async def failing_broker(_server_args, ready):
            ready.set_exception(bind_error)
            raise bind_error

        scheduler_client = MagicMock()
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.scheduler_client.run_zeromq_broker",
            failing_broker,
        )
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.scheduler_client.async_scheduler_client",
            scheduler_client,
        )
        app = FastAPI()
        app.state.server_args = SimpleNamespace(warmup_mode="off")
        entered = False

        with pytest.raises(RuntimeError, match="broker bind failed"):
            async with lifespan(app):
                entered = True

        assert not entered
        scheduler_client.initialize.assert_called_once_with(app.state.server_args)
        scheduler_client.close.assert_called_once_with()

    asyncio.run(run_test())
