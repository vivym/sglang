import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator as dg
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import ShutdownReq


class _FakeProcess:
    name = "fake-worker"

    def __init__(self):
        self.alive = True
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


class TestDiffGeneratorShutdown(unittest.TestCase):
    def test_from_server_args_sets_global_before_local_server_start(self):
        server_args = SimpleNamespace()
        events = []

        with (
            patch.object(dg, "globally_suppress_loggers"),
            patch.object(dg, "init_diffusion_tracing"),
            patch.object(dg, "set_global_server_args") as set_global,
            patch.object(dg.PortArgs, "from_server_args", return_value=object()),
            patch.object(
                DiffGenerator,
                "_start_local_server_if_needed",
                side_effect=lambda: events.append("start") or [],
            ),
            patch.object(DiffGenerator, "_run_client_warmup_if_needed"),
        ):
            set_global.side_effect = lambda args: events.append(("global", args))
            generator = DiffGenerator.from_server_args(server_args)

        self.assertIs(generator.server_args, server_args)
        self.assertEqual(events, [("global", server_args), "start"])
        self.assertTrue(generator.owns_scheduler_client)
        generator.owns_scheduler_client = False

    def test_shutdown_uses_bounded_scheduler_timeout_and_forces_worker(self):
        generator = object.__new__(DiffGenerator)
        process = _FakeProcess()
        generator.local_scheduler_process = [process]
        generator.owns_scheduler_client = True

        client = SimpleNamespace(
            forward=Mock(side_effect=TimeoutError("blocked")),
            close=Mock(),
        )

        with patch.object(dg, "sync_scheduler_client", client):
            generator.shutdown()

        request = client.forward.call_args.args[0]
        self.assertIsInstance(request, ShutdownReq)
        self.assertEqual(client.forward.call_args.kwargs, {"timeout_ms": 5000})
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertAlmostEqual(process.join_timeouts[0], 10, delta=0.1)
        self.assertEqual(process.join_timeouts[1:], [1, 1])
        self.assertIsNone(generator.local_scheduler_process)
        self.assertFalse(generator.owns_scheduler_client)
        client.close.assert_called_once_with()

    def test_del_forces_cleanup_without_scheduler_request(self):
        generator = object.__new__(DiffGenerator)
        process = _FakeProcess()
        generator.local_scheduler_process = [process]
        generator.owns_scheduler_client = True

        client = SimpleNamespace(
            forward=Mock(),
            close=Mock(),
        )

        with patch.object(dg, "sync_scheduler_client", client):
            generator.__del__()

        client.forward.assert_not_called()
        client.close.assert_called_once_with()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertIsNone(generator.local_scheduler_process)
        self.assertFalse(generator.owns_scheduler_client)

    def test_del_tolerates_missing_shutdown_globals(self):
        generator = object.__new__(DiffGenerator)
        process = _FakeProcess()
        generator.local_scheduler_process = [process]
        generator.owns_scheduler_client = True

        with (
            patch.object(dg, "logger", None),
            patch.object(dg, "sync_scheduler_client", None),
        ):
            generator.__del__()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertIsNone(generator.local_scheduler_process)
        self.assertFalse(generator.owns_scheduler_client)


if __name__ == "__main__":
    unittest.main()
