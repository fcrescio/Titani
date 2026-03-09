import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SayToUserItem:
    text: str
    retries_left: int


class ConsumerStartWaiter(Protocol):
    async def wait_consumer_started(self, timeout: float | None = None) -> bool: ...


def enqueue_say_to_user(
    queue: asyncio.Queue[SayToUserItem],
    item: SayToUserItem,
    overflow_policy: str = "drop_oldest",
) -> bool:
    policy = (overflow_policy or "drop_oldest").strip().lower()

    if queue.full():
        if policy == "drop_newest":
            return False
        if policy == "drop_oldest":
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass
        else:
            return False

    try:
        queue.put_nowait(item)
        return True
    except asyncio.QueueFull:
        return False


async def handle_say_to_user_retry(
    *,
    outbound_track: ConsumerStartWaiter,
    queue: asyncio.Queue[SayToUserItem],
    item: SayToUserItem,
    overflow_policy: str,
    retry_delay_s: float,
) -> bool:
    ok = await outbound_track.wait_consumer_started(timeout=2.0)
    if ok:
        return True

    if item.retries_left > 0:
        if retry_delay_s > 0:
            await asyncio.sleep(retry_delay_s)
        requeued = enqueue_say_to_user(
            queue,
            SayToUserItem(text=item.text, retries_left=item.retries_left - 1),
            overflow_policy=overflow_policy,
        )
        if requeued:
            logger.warning(
                "TTS consumer not started yet; say_to_user requeued (retries_left=%s)",
                item.retries_left - 1,
            )
        else:
            logger.warning("TTS consumer not started yet; requeue failed due to queue policy")
        return False

    logger.warning("TTS consumer not started yet; dropping say_to_user after retries")
    return False
