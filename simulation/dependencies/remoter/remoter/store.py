import pickle
import logging
import sqlite3
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast
import json
import uuid
from datetime import datetime

from .messages import StoreMsg, RemoterException
from .task import Task
from .persisters import persister_factory, ErrorPersister
from .async_ import async_gather
from .serialization import error_to_string, string_to_error, serialize_kwargs, deserialize_kwargs, DATETIME_FORMAT

def parse_time_from_sql(time):
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")

# ==== Storing Metadata ====


class Store:
    """Store Metadata of task results"""

    def __init__(self, path: str, remotes: List["Store"] = None, name=None) -> None:
        if name is None:
            name = uuid.uuid4().hex
        self._name = name
        self._logger = logging.getLogger(f"remoter.store.{self._name}")
        self._path = path
        self._connection = sqlite3.connect(self._path)
        self._remotes = remotes if remotes else []
        self._waiting_for_tasks = {}
        self.init_db()

    def init_db(self) -> None:
        cur = self._connection.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS tasks (task_hash BLOB NOT NULL, datetime DATETIME NOT NULL, function_module TEXT, function_name TEXT NOT NULL, function_hash BLOB NOT NULL, kwargs BLOB NOT NULL, result BLOB NOT NULL, result_type TEXT NOT NULL, failed BOOLEAN NOT NULL, time REAL NOT NULL, PRIMARY KEY(task_hash));"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS functions (function_module BLOB, function_name BLOB NOT NULL, function_hash BLOB NOT NULL, function_code BLOB NOT NULL, datetime DATETIME NOT NULL, PRIMARY KEY(function_hash));"
        )

    def _load_metadata(self, task):
        cur = self._connection.cursor()
        t = (task.calc_hash(),)
        sql = "SELECT datetime, function_module, function_name, function_hash, kwargs, result, result_type, failed, time from tasks WHERE task_hash = ?"
        r = cur.execute(sql, t).fetchone()
        cur.close()

        if r is None:
            raise RemoterException("No result in store")

        datetime_, function_module, function_name, function_hash, kwargs, result, result_type, failed, time = r
        result, result_type, failed, time, kwargs = pickle.loads(result), result_type, bool(failed), float(time), deserialize_kwargs(json.loads(kwargs), self)
        datetime_ = parse_time_from_sql(datetime_)
        msg = StoreMsg(
            datetime_,
            function_module,
            function_name,
            function_hash,
            kwargs,
            result,
            task.calc_hash(),
            result_type,
            failed,
            time,
        )
        return msg

    def _make_task_from_sqlite_return(self, r):
        task_hash, datetime_, function_module, function_name, function_hash, kwargs, result, result_type, failed, time = r
        result, result_type, failed, time, kwargs = pickle.loads(result), result_type, bool(failed), float(time), deserialize_kwargs(json.loads(kwargs), self)
        datetime_ = parse_time_from_sql(datetime_)
        persister = persister_factory(result_type)
        task = Task(persister=persister, store=self, datetime=datetime_, function_module=function_module, function_name=function_name, function_hash=function_hash, kwargs=kwargs, result=result, task_hash=task_hash, done=True, failed=failed, time=time)
        return task

    def load_from_hash(self, hash) -> 'Task':
        cur = self._connection.cursor()
        t = (hash,)
        sql = "SELECT task_hash, datetime, function_module, function_name, function_hash, kwargs, result, result_type, failed, time from tasks WHERE task_hash = ?"
        r = cur.execute(sql, t).fetchone()
        cur.close()

        if r is None:
            task = Task(persister=persister_factory("remoter.InbasePersister()"), store=self, datetime=None, function_module=None, function_name=None, function_hash=None, kwargs=None, result=None, task_hash=hash, done=False, failed=False, time=None)
        else:
            task = self._make_task_from_sqlite_return(r)
        return task

    def _insert_from_hash(self, task, do_load=True) -> 'Task':
        cur = self._connection.cursor()
        t = (task.task_hash,)
        sql = "SELECT task_hash, datetime, function_module, function_name, function_hash, kwargs, result, result_type, failed, time from tasks WHERE task_hash = ?"
        r = cur.execute(sql, t).fetchone()
        cur.close()

        if r is not None:
            task_hash, datetime_, function_module, function_name, function_hash, kwargs, result, result_type, failed, time = r
            result, result_type, failed, time, kwargs = pickle.loads(result), result_type, bool(failed), float(time), deserialize_kwargs(json.loads(kwargs), self)
            datetime_ = parse_time_from_sql(datetime_)
            persister = persister_factory(result_type)
            task.persister = persister
            task.datetime = datetime_
            task.function_module = function_module
            task.function_name = function_name
            task.function_hash = function_hash
            task.kwargs = kwargs
            task.result = result
            task.failed = failed
            task.time = time
            if do_load:
                task.load()
        return task

    def load(self, task) -> Any:
        o_task = task
        msg = self._load_metadata(task)
        persister = persister_factory(msg.result_type)
        self._logger.debug("For task %s using persister %s", task, persister)
        task.failed = msg.failed
        task = persister.from_msg(msg)
        task.store = self
        self._logger.debug("Persister returned task %s", task)
        for col in ["datetime", "function_module", "function_name", "function_hash", "kwargs", "result", "persister", "failed", "time"]:
            setattr(o_task, col, getattr(task, col))
        if msg.failed:
            o_task.failed = True
            o_task.exception = task.exception
        return task.result

    def load_all(self):
        cur = self._connection.cursor()
        sql = "SELECT task_hash, datetime, function_module, function_name, function_hash, kwargs, result, failed, time FROM tasks"
        rows = cur.execute(sql).fetchall()
        cur.close()
        tasks = [self._make_task_from_sqlite_return(r) for r in rows]
        return tasks

    def _income(self, msg):
        try:
            cur = self._connection.cursor()
            t = (
                msg.task_hash,
                msg.datetime,
                msg.function_module,
                msg.function_name,
                msg.function_hash,
                json.dumps(serialize_kwargs(msg.kwargs)),
                pickle.dumps(msg.result),
                msg.result_type,
                msg.failed,
                msg.time,
            )
            cur.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", t)
        except sqlite3.IntegrityError:
            self._logger.warning("Tried to insert a task into the store which is already present")
        cur.close()
        self._connection.commit()

        if msg.task_hash in self._waiting_for_tasks:
            for task in self._waiting_for_tasks[msg.task_hash]:
                self._insert_from_hash(task, do_load=False)
            del self._waiting_for_tasks[msg.task_hash]

    def _funcsave(self, msg):
        cur = self._connection.cursor()
        t = (
            msg.function_module,
            msg.function_name,
            msg.function_hash,
            msg.function_code,
            msg.datetime,
        )
        cur.execute("INSERT INTO functions VALUES (?, ?, ?, ?, ?)", t)
        cur.close()
        self._connection.commit()

    async def income(self, msg):
        self._income(msg)

    async def save(self, task) -> None:
        if task.failed:
            task.persister = ErrorPersister()
        msg = task.persister.to_msg(task)
        await self.income(msg)
        await async_gather(*(remote.income(msg) for remote in self._remotes))

    def _save_force(self, task) -> None:
        msg = task.persister.to_msg(task)
        self._income(msg)


class Remote(Store):
    """Forwards incoming StoreMsgs to remote stores

    Used by Workers to forward StoreMsgs to the Server"""

    def __init__(self, server, name=None):
        if name is None:
            name = uuid.uuid4().hex
        self._name = name
        self._logger = logging.getLogger(f"remoter.store.Remote.{self._name}")
        self._server = server

    async def save(self, task) -> None:
        raise NotImplemented

    def load(self, task) -> Any:
        raise NotImplemented

    async def income(self, msg) -> None:
        self._logger.debug("Forwarding")
        await self._server.send_async(msg)
        self._logger.debug("Forwarded")
        self._logger.debug("Receiving Forwarding Ack")
        await self._server.recv_async()
        self._logger.debug("Received Forwarding Ack")

