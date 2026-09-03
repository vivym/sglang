from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._batching_max_size = 2
    scheduler._batching_delay_s = 0.5
    scheduler._return_item_result = MagicMock()
    return scheduler


def test_sequential_group_logs_completion_after_returning_every_request():
    scheduler = _scheduler()
    items = [(b"first", object()), (b"second", object())]
    outputs = iter([object(), object()])

    with patch(
        "sglang.multimodal_gen.runtime.managers.scheduler.logger.info"
    ) as log_info:
        scheduler._return_results_sequentially(items, outputs)

    assert scheduler._return_item_result.call_count == 2
    log_info.assert_called_once_with(
        "Processed native grouped batch sequentially: %d/%d request(s) "
        "with max_delay=%.2fms",
        2,
        2,
        500.0,
    )


def test_sequential_group_does_not_log_completion_when_output_is_missing():
    scheduler = _scheduler()
    items = [(b"first", object()), (b"second", object())]

    with patch(
        "sglang.multimodal_gen.runtime.managers.scheduler.logger.info"
    ) as log_info:
        scheduler._return_results_sequentially(items, iter([object()]))

    assert scheduler._return_item_result.call_count == 2
    assert "fewer outputs than requests" in (
        scheduler._return_item_result.call_args_list[-1].args[1].error
    )
    log_info.assert_not_called()
