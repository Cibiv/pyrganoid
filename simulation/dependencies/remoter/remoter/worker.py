from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast
import uuid
import logging
import importlib
import inspect
import sys
import time

from . import globaladdress
from .store import Store, Remote
from .task import Task
from .async_ import async_run, async_sleep
from .messages import FreeMsg, Message, OKMsg, TaskMsg, QuitMsg, LockMsg, WaitMsg, ResultMsg, StoreMsg, FailMsg, ExceptionMsg, assert_never, parse_json_to_message, message_to_json, JoinMsg, FSaveMsg
from .persisters import persister_factory, InbasePersister
from .socket_ import ReqSocket

class Worker:

    instances: List["Worker"] = []

    def __init__(self, address: str = "ipc://.remoter.ipc", name: Optional[str] = None) -> None:
        if name:
            self._name = name
        else:
            self._name = uuid.uuid4().hex
        self._modules: dict[str, Any] = {}
        self._address: str = address
        self._socket: Optional[ReqSocket] = None
        self._running: bool = False
        self._result: Optional[Any] = None
        self._result_computed: bool = False
        self._task: Optional[Task]
        self._store = Store(":memory:", [Remote(self, name=self._name)])
        self._wait = None

        globaladdress.global_address = address

        Worker.instances.append(self)
        self.logger = logging.getLogger(f"remoter.client.{self._name}")

    def __del__(self) -> None:
        Worker.instances.remove(self)

    def start_async(self):
        async_run(self._start_async)

    async def _start_async(self) -> None:
        self._socket = sock = ReqSocket()
        self.logger.info("Connecting to %s", self._address)
        sock.connect(self._address)
        self.logger.debug("Connected successfully to %s", self._address)

        self._running = True

        try:
            while self._running:
                freemsg = FreeMsg(self._name)
                self.logger.info("Sending a message indicating I'm free")
                await self.send_async(freemsg)
                try:
                    msg = await self.recv_async()
                except TimeoutError as e:
                    self._running = False
                    self.logger.error("Did not receive an answer to my FreeMsg in time. I'm quitting without notifying the server")
                    continue
                await self.handle_message(msg)
        finally:
            self._socket.close()
        self.logger.info("Client shut down done")


    async def handle_message(self, message: Message) -> None:
        if isinstance(message, FreeMsg):
            raise Exception("Worker got a Free Message")
        elif isinstance(message, OKMsg):
            raise Exception("Worker got an OK Message")
        elif isinstance(message, TaskMsg):
            await self._handle_task(message)
        elif isinstance(message, QuitMsg):
            self.logger.info("Worker was told to quit")
            self._running = False
        elif isinstance(message, LockMsg):
            raise Exception("Worker got an Lock Message")
        elif isinstance(message, WaitMsg):
            await self._handle_wait(message)
        elif isinstance(message, ResultMsg):
            raise Exception("Worker got an Result Message")
        elif isinstance(message, StoreMsg):
            raise Exception("Worker got an Store Message")
        elif isinstance(message, FailMsg):
            raise Exception("Worker got an Fail Message")
        elif isinstance(message, ExceptionMsg):
            raise Exception("Worker got an Exception Message")
        elif isinstance(message, JoinMsg):
            raise Exception("Worker got an JoinMessage")
        elif isinstance(message, FSaveMsg):
            raise Exception("Worker got an JoinMessage")
        else:
            assert_never(message)

    async def load_kwargs(self, kwargs):
        def find_tasks(value):
            if isinstance(value, Task):
                return [value]
            elif isinstance(value, list):
                l = []
                for v in value:
                    for task in find_tasks(v):
                        l.append(task)
                return l
            elif isinstance(value, dict):
                l = []
                for v in value.values():
                    for task in find_tasks(v):
                        l.append(task)
                return l
            else:
                return []

        tasks = find_tasks(kwargs)

        for task in tasks:
            await self.send_async(task.to_task_message(request=False))
            try:
                storemsg = await self.recv_async()
                self._store._income(storemsg)
                task.load()
            except TimeoutError as e:
                self._running = False
                logging.error("Did not receive an answer to my TaskMsg, is the server done? Quitting.")
                break

    async def _handle_task(self, message):
        await self.load_kwargs(message.kwargs)
        if not self._running:
            return
        self.logger.info("Worker is executing task %s with hash %s", message.function_name, message.task_hash)
        persister = persister_factory(message.persister)
        self.logger.debug("Using persister %s for task %s", persister, message)
        task = Task(
            persister,
            self._store,
            message.datetime,
            message.function_module,
            message.function_name,
            message.function_hash,
            message.kwargs,
            result=None,
            task_hash=message.task_hash,
            done=False,
            time=None
        )
        task = await self.execute(task)
        await task.save()
        await self.send_async(task.to_result_message(self._name))
        try:
            await self.recv_async()
        except TimeoutError as e:
            logging.error("Did not receive an answer to my ResultMsg, is the server done? Continuing anyway.")


    async def _handle_wait(self, message):
        return await async_sleep(message.time)

    async def handle_exception_async(self, e) -> None:
        #await self.send_async(ExceptionMsg(self._name, json.dumps(tblib.Traceback(e[2]).to_dict())))
        await self.send_async(ExceptionMsg(self._name, str(e)))
        try:
            await self.recv_async()  # TODO check result was okay
        except TimeoutError as e:
            logging.error("Did not receive an answer to my ExceptionMsg, is the server done? Continuing anyway.")

    def _load_value(self, value):
        if isinstance(value, Task):
            return value.load()
        elif isinstance(value, list):
            return [self._load_value(v) for v in value]
        elif isinstance(value, dict):
            return {key: self._load_value(v) for key, v in value.items()}
        else:
            return value


    async def execute(self, task: Task) -> Task:
        if task.function_module not in self._modules:
            self._modules[task.function_module] = importlib.import_module(task.function_module)
        f = getattr(self._modules[task.function_module], task.function_name)
        self.logger.debug("Executing task")
        start_time = time.time()
        end_time = None
        kwargs = self._load_value(task.kwargs)
        try:
            start_time = time.time()
            result_or_awaitable = f(**kwargs)
            end_time = time.time()
            if inspect.isawaitable(result_or_awaitable):
                start_time = time.time()
                result = await result_or_awaitable
                end_time = time.time()
            else:
                result = result_or_awaitable
            task.done = True
            task.result = result
            task.task_hash = task.calc_hash()
            task.time = end_time - start_time
        except Exception as e:
            if end_time is None:
                end_time = time.time()
            self.logger.exception("Client encountered an exception when executing a task.")
            await self.handle_exception_async(e)
            task.failed = True
            task.done = True
            task.time = end_time - start_time
            err = sys.exc_info()
            task.exception = (err[1], err[2]) if err[1] and err[2] else None
            p = InbasePersister()
            task.persister = p
        return task

    async def recv_async(self) -> Message:
        if self._socket is not None:
            self.logger.debug("Waiting for an answer")
            msg_json = await self._socket.async_recv_json()
            msg = parse_json_to_message(msg_json, self._store)
            return msg
        else:
            raise Exception("Worker has to connect before being able to receive messages")

    async def send_async(self, msg: Message) -> None:
        if self._socket:
            self.logger.debug("Sending msg %s", msg)
            j = message_to_json(msg)
            await self._socket.async_send_json(j)
        else:
            raise Exception("Worker has to connect before being able to send messages")

