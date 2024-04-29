from __future__ import annotations

import inspect
from hashlib import sha256
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast
import logging
from datetime import datetime
from types import TracebackType

from .messages import ResultMsg, TaskMsg, FSaveMsg, RemoterException
from .serialization import serialize_kwargs

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .store import Store
    from .persisters import Persister

# ==== Task ====


def depends(*fs):
    def wrapper(f):
        if not hasattr(f, "_remoter_depends"):
            f._remoter_depends = []
        for dep in fs:
            f._remoter_depends.append(dep)
        return f
    return wrapper


def hash_function(f):
    """Hash the source of the function f"""
    source = inspect.getsource(f)
    s = sha256(source.encode("utf-8"))
    if hasattr(f, "_remoter_depends"):
        for x in f._remoter_depends:
            s.update(hash_function(x).encode("utf8"))
    return s.hexdigest()


@dataclass
class Task:
    persister: Optional[Persister]
    store: Store
    datetime: datetime
    function_module: str
    function_name: str
    function_hash: str
    kwargs: Dict[str, Any]
    result: Optional[Any] = None
    task_hash: Optional[str] = None
    done: bool = False
    ignore_hash = False
    loaded = False
    exception: Optional[tuple[BaseException, TracebackType]] = None
    failed: bool = False
    time: Optional[float] = None
    code: Optional[str] = None

    def calc_hash(self):
        if self.task_hash is None:
            s = sha256()
            s.update(self.function_module.encode("utf8"))
            s.update(self.function_name.encode("utf8"))
            s.update(serialize_kwargs(self.kwargs).encode("utf8"))
            if not self.ignore_hash:
                s.update(self.function_hash.encode("utf8"))
            self.task_hash = s.hexdigest()
        return self.task_hash

    def to_result_message(self, worker_name: str) -> ResultMsg:
        assert self.task_hash
        return ResultMsg(
            task_hash=self.task_hash,
            datetime=self.datetime,
            function_module=self.function_module,
            function_name=self.function_name,
            kwargs=self.kwargs,
            result=None,
            worker_name=worker_name,
        )

    def to_task_message(self, request=True) -> TaskMsg:
        self.calc_hash()
        assert self.task_hash
        return TaskMsg(
            persister=self.persister.path if self.persister else None,
            task_hash=self.task_hash,
            datetime=self.datetime,
            function_module=self.function_module,
            function_name=self.function_name,
            function_hash=self.function_hash,
            kwargs=self.kwargs,
            failed=self.failed,
            exception=None,
            request=request,
            time=self.time
        )

    @classmethod
    def from_task_message(cls, message, persister, store) -> 'Task':
        task = Task(
            persister,
            store,
            message.datetime,
            message.function_module,
            message.function_name,
            message.function_hash,
            message.kwargs,
            result=None,
            task_hash=message.task_hash,
            done=False,
            time=message.time
        )
        return task

    def to_fsave_message(self):
        return FSaveMsg(
            datetime=self.datetime,
            function_module=self.function_module,
            function_name=self.function_name,
            function_hash=self.function_hash,
            function_code=self.code
        )

    def load(self):
        if self.loaded:
            return self.result
        self.calc_hash()
        logging.debug(f"Trying to load task {self}")
        try:
            r = self.store.load(self)
            self.result = r
            self.loaded = True
            self.done = True
        except RemoterException as e:
            logging.debug("Loading of task failed %s", e)
            return False
        return self.result

    def load_metadata(self):
        try:
            msg = self.store._load_metadata(self)
        except RemoterException:
            return False
        return msg

    def is_loadable(self):
        try:
            self.store._load_metadata(self)
        except RemoterException:
            return False
        return True

    async def save(self):
        r = await self.store.save(self)

    async def wait(self):
        if not self.loaded:
            self.load()
        return self.result

    def _save_force(self):
        self.store._save_force(self)

    def __str__(self):
        kwargs = str(self.kwargs)
        if len(kwargs) > 20:
            kwargs = kwargs[:10] + "..." + kwargs[-10:]
        return f"Task({self.task_hash}, {self.function_module}, {self.function_name}, {self.function_hash}, {kwargs}, result=..., done={self.done})"

    def __repr__(self):
        kwargs = str(self.kwargs)
        if len(kwargs) > 20:
            kwargs = kwargs[:10] + "..." + kwargs[-10:]
        return f"Task({self.task_hash}, done={self.done})"

    def __eq__(self, o):
        if not isinstance(o, Task):
            return False
        return self.calc_hash() == o.calc_hash()

    def __hash__(self):
        v = int(self.calc_hash(), 16)
        return v
