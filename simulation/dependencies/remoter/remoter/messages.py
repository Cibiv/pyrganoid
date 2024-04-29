from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast
from types import TracebackType
import pickle
import base64

from .serialization import error_to_string, string_to_error, serialize_kwargs, deserialize_kwargs, DATETIME_FORMAT


try:
    from typing import Literal
except:

    class MyLiteral:
        def __init__(self, t):
            self.t = t

        def __getitem__(self, *args, **kwargs):
            return self.t

    Literal = MyLiteral(str)  # type: ignore


class RemoterException(Exception):
    pass


## ==== Messages ====
#
# All Messages that can be exchanged by workers and the server


@dataclass
class FreeMsg:
    """Indicates the worker is free for more work"""

    name: str


@dataclass
class OKMsg:
    """Indicates the requested action succeeded

    Used as answer for Lock, Result and Store messages."""

    pass


@dataclass
class FailMsg:
    """Indicates the requested action failed terminally

    Used as answer for Lock, Result and Store messages."""

    name: str
    message: str


@dataclass
class TaskMsg:
    """Indicates a task to be executed by the worker"""

    persister: str
    task_hash: str
    datetime: datetime
    function_module: str
    function_name: str
    function_hash: str
    kwargs: Dict[str, Any]
    failed: bool
    exception: Optional[tuple[BaseException, TracebackType]]
    request: bool
    time: Optional[float]

@dataclass
class FSaveMsg:
    """Saving a function"""
    datetime: datetime
    function_module: str
    function_name: str
    function_hash: str
    function_code: str

@dataclass
class ResultMsg:
    """Indicates the result of the assigned task.

    outdated, currently `Store`s communicate via StoreMsgs"""

    task_hash: str
    datetime: datetime
    function_name: str
    function_module: str
    kwargs: Dict[str, Any]
    worker_name: str
    result: Any = None


@dataclass
class WaitMsg:
    """Indicate the worker that no work is available, but more might follow

    Is used as an answer to a `FreeMsg` or to a `LockMsg`"""

    time: int


@dataclass
class QuitMsg:
    """Indicate the worker to shut down

    is used as an answer to a `FreeMsg`"""

    name: str
    client_type: str

@dataclass
class LockMsg:
    """Asks to acquire a lock"""

    operation: Literal["lock", "unlock"]
    lock: str
    name: str


@dataclass
class StoreMsg:
    """Updates a Store"""

    datetime: datetime
    function_module: str
    function_name: str
    function_hash: str
    kwargs: Dict[str, Any]
    result: Union[Any, bytes]
    task_hash: str
    result_type: str
    failed: bool
    time: float

    def __repr__(self):
        kwargs = str(self.kwargs)
        if len(kwargs) > 20:
            kwargs = kwargs[:10] + "..." + kwargs[-10:]

        return f"StoreMsg(datetime={self.datetime}, function_module={self.function_module}, function_name={self.function_name}, function_hash={self.function_hash}, kwargs={kwargs}, result=..., task_hash={self.task_hash}, result_type={self.result_type}, failed={self.failed}, time={self.time})"


    def __str__(self):
        kwargs = str(self.kwargs)
        if len(kwargs) > 20:
            kwargs = kwargs[:10] + "..." + kwargs[-10:]

        return f"StoreMsg(datetime={self.datetime}, function_module={self.function_module}, function_name={self.function_name}, function_hash={self.function_hash}, kwargs={kwargs}, result=..., task_hash={self.task_hash}, result_type={self.result_type}, failed={self.failed}, time={self.time})"



@dataclass
class ExceptionMsg:
    name: str
    exception: str

@dataclass
class JoinMsg:
    name: str
    client_type: str

# Typing Stuff to make sure all messages are always handled in all code paths

Message     =          Union[ OKMsg , FreeMsg, TaskMsg, QuitMsg, LockMsg, WaitMsg, ResultMsg, StoreMsg, FailMsg, ExceptionMsg, JoinMsg, FSaveMsg]

# A serialized message indicates its type by a token, this lists the available tokens.
# Order is the same as the Message Type
MessageType =               ["OKOK" , "FREE" ,  "TASK", "QUIT",  "LOCK",  "WAIT",  "RESULT",  "STORE",  "FAIL", "EXCEPTION", "JOIN", "FSAVE"]
# Type of MesssageType, used when only the tokens listed in MessageType are accectable
MessageTypeLiteral = Literal["OKOK" , "FREE" ,  "TASK", "QUIT",  "LOCK",  "WAIT",  "RESULT",  "STORE",  "FAIL", "EXCEPTION", "JOIN", "FSAVE"]

# Can be used to enable exhaustive type checking in mypy, see https://hakibenita.com/python-mypy-exhaustive-checking
def assert_never(x: NoReturn) -> NoReturn:
    raise Exception("This should never happen.")


# ==== De/Serializing Messages ====


def parse_json_to_message(j: Dict[str, Any], store: 'Store') -> Message:
    """Deserialize a json-like dict to a message"""
    if "type" not in j:
        raise Exception("Message had no declared type.")
    t: MessageTypeLiteral = j["type"]
    if t not in MessageType:
        raise Exception(f"Unknown type {t}")
    if t == "FREE":
        if "name" not in j:
            raise Exception("Free message does not contain worker name")
        return FreeMsg(j["name"])
    elif t == "OKOK":
        return OKMsg()
    elif t == "FSAVE":
        return FSaveMsg(
            datetime=datetime.strptime(j["datetime"], DATETIME_FORMAT) if j['datetime'] else None,
            function_module=j["function_module"],
            function_name=j["function_name"],
            function_hash=j["function_hash"],
            function_code=j["function_code"],
        )
    elif t == "TASK":
        return TaskMsg(
            persister=j["persister"],
            task_hash=j["task_hash"],
            datetime=datetime.strptime(j["datetime"], DATETIME_FORMAT) if j['datetime'] else None,
            function_module=j["function_module"],
            function_name=j["function_name"],
            function_hash=j["function_hash"],
            kwargs=deserialize_kwargs(j["kwargs"], store),
            failed=j["failed"],
            exception=string_to_error(j["exception"]),
            request=j["request"],
            time=j["time"]
        )
    elif t == "QUIT":
        return QuitMsg(name=j["name"], client_type=j["client_type"])
    elif t == "LOCK":
        return LockMsg(operation=j["op"], lock=j["lock"], name=j["name"])
    elif t == "WAIT":
        return WaitMsg(j["time"])
    elif t == "RESULT":
        return ResultMsg(
            task_hash=j["task_hash"],
            datetime=datetime.strptime(j["datetime"], DATETIME_FORMAT),
            function_module=j["function_module"],
            function_name=j["function_name"],
            kwargs=deserialize_kwargs(j["kwargs"], store),
            worker_name=j["worker_name"],
            result=j["result"],
        )
    elif t == "STORE":
        r = pickle.loads(base64.b64decode(j["result"].encode("utf-8")))
        return StoreMsg(
            datetime=datetime.strptime(j["datetime"], DATETIME_FORMAT),
            function_module=j["function_module"],
            function_name=j["function_name"],
            function_hash=j["function_hash"],
            kwargs=deserialize_kwargs(j["kwargs"], store),
            result=r,
            task_hash=j["task_hash"],
            result_type=j["result_type"],
            failed=j["failed"],
            time=j["time"],
        )
    elif t == "FAIL":
        return FailMsg(name=j["name"], message=j["message"])
    elif t == "EXCEPTION":
        return ExceptionMsg(name=j["name"], exception=j["exception"])
    elif t == "JOIN":
        return JoinMsg(name=j["name"], client_type=j["client_type"])
    else:
        assert_never(t)


def message_to_json(msg: Message) -> Dict[str, Any]:
    """Serialize a message to json-like dict"""
    if isinstance(msg, OKMsg):
        return {"type": "OKOK"}
    elif isinstance(msg, FreeMsg):
        return {"type": "FREE", "name": msg.name}
    elif isinstance(msg, TaskMsg):
        return {
            "type": "TASK",
            "persister": msg.persister,
            "task_hash": msg.task_hash,
            "datetime": msg.datetime.strftime(DATETIME_FORMAT) if msg.datetime else None,
            "function_module": msg.function_module,
            "function_name": msg.function_name,
            "function_hash": msg.function_hash,
            "kwargs": serialize_kwargs(msg.kwargs),
            "failed": msg.failed,
            "exception": error_to_string(msg.exception),
            "request": msg.request,
            "time": msg.time
        }
    elif isinstance(msg, FSaveMsg):
        return {
            "type": "FSAVE",
            "datetime": msg.datetime.strftime(DATETIME_FORMAT) if msg.datetime else None,
            "function_module": msg.function_module,
            "function_name": msg.function_name,
            "function_hash": msg.function_hash,
            "function_code": msg.function_code,
        }
    elif isinstance(msg, QuitMsg):
        return {"type": "QUIT", "name": msg.name, "client_type": msg.client_type}
    elif isinstance(msg, LockMsg):
        return {"type": "LOCK", "op": msg.operation, "name": msg.name, "lock": msg.lock}
    elif isinstance(msg, WaitMsg):
        return {"type": "WAIT", "time": msg.time}
    elif isinstance(msg, ResultMsg):
        return {
            "type": "RESULT",
            "task_hash": msg.task_hash,
            "datetime": msg.datetime.strftime(DATETIME_FORMAT),
            "function_module": msg.function_module,
            "function_name": msg.function_name,
            "kwargs": serialize_kwargs(msg.kwargs),
            "worker_name": msg.worker_name,
            "result": msg.result,
        }
    elif isinstance(msg, StoreMsg):
        r = base64.b64encode(pickle.dumps(msg.result)).decode("utf-8")
        return {
            "datetime": msg.datetime.strftime(DATETIME_FORMAT),
            "type": "STORE",
            "function_module": msg.function_module,
            "function_name": msg.function_name,
            "function_hash": msg.function_hash,
            "kwargs": serialize_kwargs(msg.kwargs),
            "result": r,
            "task_hash": msg.task_hash,
            "result_type": msg.result_type,
            "failed": msg.failed,
            "time": msg.time,
        }
    elif isinstance(msg, FailMsg):
        return {"type": "FAIL", "name": msg.name, "message": msg.message}
    elif isinstance(msg, ExceptionMsg):
        return {"type": "EXCEPTION", "name": msg.name, "exception": msg.exception}
    elif isinstance(msg, JoinMsg):
        return {"type": "JOIN", "name": msg.name, "client_type": msg.client_type}
    else:
        assert_never(msg)


