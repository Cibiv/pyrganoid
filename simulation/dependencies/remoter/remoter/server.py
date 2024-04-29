from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast
import os.path
import sys
import inspect
from datetime import datetime
import time
import logging
import traceback

from .persisters import PersisterMapper, persister_factory
from .task import Task, hash_function
from .store import Store, Remote
from .async_ import async_run, async_gather
from .messages import parse_json_to_message, message_to_json, Message, Message, FreeMsg, QuitMsg, WaitMsg, OKMsg, TaskMsg, LockMsg, ResultMsg, StoreMsg, FailMsg, ExceptionMsg, assert_never, JoinMsg, FSaveMsg
from .socket_ import RepSocket, PullSocket
from .worker import Worker

DEFAULT_STORE_PATH = ".remoter.sqlite3"

class WorkerManager:

    def __init__(self):
        self.workers = []

        self._working_on = {}
        self._tasks_in_progress_list: List[Task] = []

    @property
    def tasks_in_progress(self):
        return self._tasks_in_progress_list

    def __len__(self):
        return len(self.workers)

    def remove(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)
        if worker in self._working_on:
            del self._working_on[worker]

    def add(self, worker):
        if worker not in self.workers:
            self.workers.append(worker)

    def notice(self, worker):
        self.add(worker)

    def works_on(self, worker, task):
        self.notice(worker)
        self._working_on[worker] = task
        self._tasks_in_progress_list.append(task)

    def get_working_on(self, worker):
        return self._working_on.get(worker, None)

    def done(self, worker):
        if worker in self._working_on:
            task = self._working_on[worker]
            del self._working_on[worker]
            self._tasks_in_progress_list.remove(task)


class TaskQueue:

    def __init__(self):
        self._ready_tasks: List[Task] = []
        self._logger = logging.getLogger("remoter.server.TaskQueue")

        self._tasks_scheduled_ever = set()
        self._total_tasks = 0

        self._tasks_being_waited_on = {}
        self._task_waiting_counter = {}


    def _find_dependent_tasks(self, v):
        if isinstance(v, dict):
            l = []
            for value in v.values():
                for t in self._find_dependent_tasks(value):
                    l.append(t)
            return list(set(l))
        elif isinstance(v, list):
            l = []
            for value in v:
                for t in self._find_dependent_tasks(value):
                    l.append(t)
            return list(set(l))
        elif isinstance(v, Task):
            if v.done:
                return []
            else:
                return [v]
        else:
            return []

    def add(self, task: Task) -> None:
        if task.calc_hash() in self._tasks_scheduled_ever:
            return
        self._tasks_scheduled_ever.add(task.calc_hash())
        self._total_tasks += 1
        tasks_to_wait_for = self._find_dependent_tasks(task.kwargs)
        for twf in tasks_to_wait_for:
            if twf.calc_hash() not in self._tasks_being_waited_on:
                self._tasks_being_waited_on[twf.calc_hash()] = []
            if task not in self._tasks_being_waited_on[twf.calc_hash()]:
                self._tasks_being_waited_on[twf.calc_hash()].append(task)
        if len(tasks_to_wait_for) > 0:
            self._logger.info("Task with hash %s needs to wait for %s many tasks", task.calc_hash(), len(tasks_to_wait_for))
            self._task_waiting_counter[task.calc_hash()] = len(tasks_to_wait_for)
        else:
            self._ready_tasks.append(task)

    def done(self, task_hash):
        if task_hash in self._tasks_being_waited_on:
            for waiting_task in self._tasks_being_waited_on[task_hash]:
                counter = self._task_waiting_counter.get(waiting_task.calc_hash(), 0) - 1
                if counter <= 0:
                    self._logger.debug("Task with hash %s has counter %s and is now being considered for tasks", waiting_task.calc_hash(), counter)
                    self._ready_tasks.append(waiting_task)
                    del self._task_waiting_counter[waiting_task.calc_hash()]
                else:
                    self._task_waiting_counter[waiting_task.calc_hash()] = counter
            del self._tasks_being_waited_on[task_hash]

    def failed(self, task_hash):
        self._ready_tasks = []

    def remove(self, task: Task) -> bool:
        if not task in self._ready_tasks:
            return False
        self._ready_tasks.remove(task)
        return True

    def get_ready_task(self):
        if len(self._ready_tasks) == 0:
            return None
        return self._ready_tasks.pop(0)

    def __len__(self):
        return len(self._ready_tasks) + len(self._task_waiting_counter)


class LoggingServer:

    def __init__(self, address):
        self._address = "tcp://127.0.0.1:12345"
        self._socket = PullSocket()
        self._running = False

    async def serve_logging_async(self) -> None:
        logger = logging.getLogger("remoter.logger")
        logger.debug("Starting logging server")
        logger.info("Logging server listening on %s", self._address)
        self._running = True
        self._socket.listen(self._address)
        out_logger = logging.getLogger()

        try:
            logger.debug("Starting to log, status is %s", self._running)
            while self._running:
                d = await self._socket.async_recv_json()
                logger.debug("Logging something")
                record = logging.makeLogRecord(d)
                out_logger.handle(record)
        finally:
            self._socket.close()
            logger.info("Logging server shut down done")



class Server:

    instance: Optional["Server"] = None

    def __init__(
        self,
        address: str = "ipc://.remoter.ipc",
        logging_address: str = "ipc://.remoter.logging.ipc",
        storepath: str = DEFAULT_STORE_PATH,
        persister_mapper=PersisterMapper,
        enable_logging=False,
        launch_workers=None,
        workers=None,
        lazy=False,
        client_hang_around=True,
        server_hang_around=True,
        wait_time=1,
    ):

        self._logger = logging.getLogger("remoter.server")
        self._schedulers: Set[str] = set()
        self._scheduler_remotes: dict[str, "Remote"] = {}
        self._address = address
        self._socket: Optional[RepSocket] = None
        self._locks: Dict[str, str] = {}
        self._running = False
        self._store = Store(storepath, name="server_store")
        self._persister_mapper = persister_mapper
        self._logging = enable_logging
        self._failed = False
        self._failing_messages: list[str] = []
        self._current_file_module = os.path.split(sys.argv[0])[-1].replace(".py", "")
        self._logging_address = logging_address
        self._launch_worker_mode = launch_workers
        self._launch_worker_number = workers
        self._total_tasks = 0
        self._lazy = lazy
        self._client_exit_on_free = not client_hang_around
        self._server_exit_on_free = not server_hang_around
        self._worker = None
        self._wait_time = wait_time
        self._should_run = False

        self.task_queue = TaskQueue()
        self.worker_manager = WorkerManager()

        worker_modes = [None, "async"]
        if launch_workers not in worker_modes:
            raise ValueError(f"Launch worker must be from {worker_modes}")
        Server.instance = self

    def stop(self):
        self._logger.info("Stopping server, waiting for all clients to quit")
        self._server_exit_on_free = True
        self._client_exit_on_free = True

    def fullfill(self, task, result):
        task.result = result
        task.done = True
        task.time = -1
        task._save_force()
        self.task_queue.done(task.task_hash)

    def create_task(self, f: Callable, *args: Any, **kwargs: Any) -> Task:
        params = inspect.signature(f).parameters
        kwargs.update({param: arg for param, arg in zip(params, args)})
        module = f.__module__ if f.__module__ != "__main__" else self._current_file_module
        task = Task(None, self._store, datetime.now(), module, f.__name__, hash_function(f), kwargs)
        task.persister = self._persister_mapper(f)
        task.task_hash = task.calc_hash()
        return task

    def run(self, f: Callable, *args: Any, **kwargs: Any) -> Task:
        task = self.create_task(f, *args, **kwargs)

        if not task.is_loadable():
            self.task_queue.add(task)
        else:
            task.load()
        return task

    def start_async(self) -> None:
        if self._worker:
            async_run(self._start_server_and_worker)
        else:
            async_run(self._start_async)

    async def _start_server_and_worker(self):
        return await async_gather(self._worker._start_async, self._start_async)

    async def _start_async(self) -> None:
        self._should_run = True
        self._running = True
        tasks = [self.serve_async]
        if self._logging:
            raise NotImplementedError()
        await self.serve_async()
        self._running = False
        self._logger.info("Server shut down done")

    async def serve_async(self) -> None:
        logger = self._logger
        self._socket = sock = RepSocket()
        logger.info("Listening on %s", self._address)
        sock.listen(self._address)

        _number_tasks = 0
        _number_workers = 0
        _number_schedulers = 0

        try:
            logger.debug("Serving")
            while self._should_run:

                if (len(self.task_queue) != _number_tasks) or (len(self.worker_manager) != _number_workers) or (len(self._schedulers) != _number_schedulers):
                    logger.info(f"Total Tasks: %s, Tasks: %s, Workers: %s, Schedulers %s", self._total_tasks, len(self.task_queue), len(self.worker_manager), len(self._schedulers))
                    _number_tasks = len(self.task_queue)
                    _number_workers = len(self.worker_manager)
                    _number_schedulers = len(self._schedulers)

                logger.debug("Waiting for messages")
                try:
                    msg = await self.recv_async()
                except TimeoutError:
                    continue
                logger.debug("Received message: %s", msg)
                answer = self.handle_message(msg)
                logger.debug("Sending answer: %s", answer)
                await self.send_async(answer)
                self._output_status()

                if self._server_exit_on_free:
                    self._should_run = len(self.worker_manager) > 0 or len(self.task_queue) > 0 or len(self._schedulers) > 0 or len(self.worker_manager.tasks_in_progress) > 0
            logger.info("Server shutting down")
        finally:
            sock.close()
            self._running = False
        logger.info("Server Serving done")

    async def recv_async(self) -> Message:
        assert self._socket
        j = await self._socket.async_recv_json()
        return parse_json_to_message(j, self._store)

    async def send_async(self, msg: Message) -> None:
        j = message_to_json(msg)
        if self._socket:
            await self._socket.async_send_json(j)
        else:
            raise Exception("Can't send a message uninitialized.")

    def handle_message(self, message: Message) -> Message:
        if isinstance(message, FreeMsg):
            if self._failed:
                self.worker_manager.remove(message.name)
                self._logger.info("Worker %s is free, telling it to quit as we are in failed state.", message.name)
                return QuitMsg("server", "server")

            task = self.task_queue.get_ready_task()
            if task:
                self.worker_manager.works_on(message.name, task)
                self._logger.info("Worker %s is free, telling it to work on the task '%s'", message.name, task.function_name)
                return task.to_task_message()

            elif self._client_exit_on_free:
                self.worker_manager.remove(message.name)
                self._logger.info("Worker %s is free, telling it to quit as there is no more work to do", message.name)
                return QuitMsg("server", "server")
            else:
                self.worker_manager.notice(message.name)
                return WaitMsg(self._wait_time)

        elif isinstance(message, OKMsg):
            raise Exception("Server got a OkMessage")

        elif isinstance(message, TaskMsg):
            # Scheduler schedules a task
            if self._failed:
                self._logger.info("Scheduler is asking for a task, telling it to quit as we are in failed state.")
                return QuitMsg("server", "server")

            persister = persister_factory(message.persister)
            task = Task.from_task_message(message, persister, self._store)
            storemsg = task.load_metadata()
            # Task is already calculated, returning result
            if storemsg:
                self._logger.info("Someone asked for task, returning task with hash '%s'", task.task_hash)
                return storemsg
            # Task needs to be run, scheduler has to wait
            else:
                if message.request:
                    self._logger.debug("Someone asked for task, scheduling '%s'", task.function_name)
                    self.task_queue.add(task)
                return WaitMsg(self._wait_time)

        elif isinstance(message, QuitMsg):
            self._logger.info("Scheduler %s quit", message.name)
            self._schedulers.remove(message.name)
            self._store._remotes.remove(self._scheduler_remotes[message.name])
            if len(self._schedulers) == 0:
                self._logger.info("No more schedulers active, stopping")
                self.stop()
            return OKMsg()

        elif isinstance(message, LockMsg):
            if message.operation == "lock":
                if message.lock not in self._locks:
                    self._locks[message.lock] = message.name
                    return OKMsg()
                else:
                    return WaitMsg(time=1)
            else:
                if message.lock in self._locks:
                    del self._locks[message.lock]
                return OKMsg()

        elif isinstance(message, ResultMsg):
            self.worker_manager.done(message.worker_name)
            self.task_queue.done(message.task_hash)
            return OKMsg()

        elif isinstance(message, WaitMsg):
            raise Exception("Server got a WaitMsg")

        elif isinstance(message, StoreMsg):
            self._store._income(message)
            return OKMsg()

        elif isinstance(message, FailMsg):
            self._failed = True
            self._failing_messages.append(message.message)
            task = self.worker_manager.get_working_on(message.name)
            self.task_queue.failed(task.task_hash)
            self.worker_manager.remove(message.name)
            self._logger.info("Client %s informed me of failure, telling it to quit", message.name)
            return QuitMsg("server", "server")

        elif isinstance(message, ExceptionMsg):
            exception = message.exception
            logging.error(exception)
            self._failing_messages.append(exception)
            self._logger.info("Client %s informed me of an exception", message.name)
            return OKMsg()

        elif isinstance(message, JoinMsg):
            self._schedulers.add(message.name)
            remote = Remote(self, message.name)
            self._scheduler_remotes[message.name] = remote
            self._store._remotes.append(remote)
            self._logger.info("Scheduler '%s' joined", message.name)
            return OKMsg()

        elif isinstance(message, FSaveMsg):
            self._store._funcsave(message)
            return OKMsg()
        else:
            assert_never(message)

    def _launch_workers(self):
        if len(self._tasks) > 0 and self._launch_worker_mode == "async":
            self._worker = Worker(self._address, "w1")

    def _output_status(self):
        status = f"Workers: {len(self.worker_manager)}, Tasks: {self._total_tasks}, Todo: {len(self.task_queue)}"
        #print("\r" + status, end="", flush=True)


