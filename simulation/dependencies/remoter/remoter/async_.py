import sys
from typing import Coroutine, Callable, Any
from collections.abc import Iterable
import trio

async def gather(*tasks):

    async def collect(index, task, results):
        print(task)
        results[index] = await task

    results = {}
    async with trio.open_nursery() as nursery:
        for index, task in enumerate(tasks):
            nursery.start_soon(collect, index, task, results)
    return [results[i] for i in range(len(tasks))]


async_run = trio.run
async_sleep = trio.sleep
async_gather = gather

