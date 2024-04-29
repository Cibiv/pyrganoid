import uuid
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast
import logging
import time

from .messages import message_to_json, LockMsg, parse_json_to_message, message_to_json, OKMsg, WaitMsg
from .worker import Worker
from .socket_ import ReqSocket, Socket



# ==== Lock ====


class Lock:
    def __init__(self, name: str) -> None:
        self.lock = name
        self._name = uuid.uuid4().hex
        self._socket: Optional[ReqSocket] = None
        self.logger = logging.getLogger(f"remoter.lock.{name}")

    async def _get_sock_async(self) -> Socket:
        if not self._socket:

            if len(Worker.instances) == 0:
                raise Exception("Worker not instantiated")
            worker = Worker.instances[-1]
            if not worker._address:
                raise Exception("Worker context not instantiated")
            address = worker._address
            self._socket = ReqSocket()
            self.logger.debug("Connecting to: %s", address)
            self._socket.connect(address)  # type: ignore

        return self._socket  # type: ignore

    async def __aenter__(self) -> None:

        sock = await self._get_sock_async()
        msg = LockMsg("lock", self.lock, self._name)
        j = message_to_json(msg)

        while True:
            self.logger.debug("Trying to get lock: %s", j)
            await sock.async_send_json(j)
            self.logger.debug("Send json succesfully")
            j_answer = await sock.async_recv_json()
            answer = parse_json_to_message(j_answer, None)
            if isinstance(answer, OKMsg):
                self.logger.debug("Got lock")
                break
            elif isinstance(answer, WaitMsg):
                self.logger.debug("Lock not free, waiting.")
                time.sleep(answer.time)
            else:
                raise Exception("Got unexpected message on lock")

    async def __aexit__(self, type: Any, value: Any, tb: Any) -> None:
        sock = await self._get_sock_async()
        msg = LockMsg("unlock", self.lock, self._name)
        j = message_to_json(msg)

        while True:
            await sock.async_send_json(j)
            j_answer = await sock.async_recv_json()
            answer = parse_json_to_message(j_answer, None)
            if isinstance(answer, OKMsg):
                break
            elif isinstance(answer, WaitMsg):
                time.sleep(answer.time)
            else:
                raise Exception("Got unexpected message on lock")

