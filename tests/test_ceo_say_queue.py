import asyncio
import unittest

from titani.say_queue import SayToUserItem, enqueue_say_to_user, handle_say_to_user_retry


class _FakeOutboundTrack:
    def __init__(self, ready: bool):
        self.ready = ready

    async def wait_consumer_started(self, timeout: float | None = None) -> bool:
        return self.ready


class TestCeoSayQueue(unittest.IsolatedAsyncioTestCase):
    async def test_handle_say_to_user_text_requeue_when_consumer_not_ready(self) -> None:
        queue: asyncio.Queue[SayToUserItem] = asyncio.Queue(maxsize=2)
        ok = await handle_say_to_user_retry(
            outbound_track=_FakeOutboundTrack(ready=False),
            queue=queue,
            item=SayToUserItem(text="ciao", retries_left=1),
            overflow_policy="drop_oldest",
            retry_delay_s=0,
        )
        self.assertFalse(ok)
        self.assertEqual(queue.qsize(), 1)
        queued = queue.get_nowait()
        self.assertEqual(queued.text, "ciao")
        self.assertEqual(queued.retries_left, 0)

    async def test_handle_say_to_user_text_ready_consumer(self) -> None:
        queue: asyncio.Queue[SayToUserItem] = asyncio.Queue(maxsize=1)
        ok = await handle_say_to_user_retry(
            outbound_track=_FakeOutboundTrack(ready=True),
            queue=queue,
            item=SayToUserItem(text="ok", retries_left=2),
            overflow_policy="drop_oldest",
            retry_delay_s=0,
        )
        self.assertTrue(ok)
        self.assertEqual(queue.qsize(), 0)

    def test_bounded_queue_drop_oldest_policy(self) -> None:
        queue: asyncio.Queue[SayToUserItem] = asyncio.Queue(maxsize=2)
        self.assertTrue(enqueue_say_to_user(queue, SayToUserItem("a", 0), overflow_policy="drop_oldest"))
        self.assertTrue(enqueue_say_to_user(queue, SayToUserItem("b", 0), overflow_policy="drop_oldest"))
        self.assertTrue(enqueue_say_to_user(queue, SayToUserItem("c", 0), overflow_policy="drop_oldest"))

        first = queue.get_nowait()
        second = queue.get_nowait()
        self.assertEqual(first.text, "b")
        self.assertEqual(second.text, "c")

    def test_bounded_queue_drop_newest_policy(self) -> None:
        queue: asyncio.Queue[SayToUserItem] = asyncio.Queue(maxsize=1)
        self.assertTrue(enqueue_say_to_user(queue, SayToUserItem("a", 0), overflow_policy="drop_newest"))
        self.assertFalse(enqueue_say_to_user(queue, SayToUserItem("b", 0), overflow_policy="drop_newest"))
        only = queue.get_nowait()
        self.assertEqual(only.text, "a")


if __name__ == "__main__":
    unittest.main()
