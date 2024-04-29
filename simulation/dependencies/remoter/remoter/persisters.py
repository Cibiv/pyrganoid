import re
import ast
from abc import ABC, ABCMeta, abstractmethod
import os.path
import sys
import importlib
import json

from .task import Task
from .task import serialize_kwargs
from .messages import StoreMsg
from .serialization import error_to_string, string_to_error

# ==== Persisting ====
#
# Loading and Saving results
#
# -- registering persisters --
#
# Each persister is identified by a path. To figure out which persister should be used to save/load a result,
# each task has a path to a persister. The Persister_factory takes the path and gives the appropriate
# persister back. Persisters can also be defined by endusers and are automatically registered.


_persisters = {}


def register_persister(persister, path):
    _persisters[path] = persister


def persister_factory(path):
    """Return the persister registered via path"""
    m = re.match("(?P<name>.+)\\((?P<args>.*)\\)", path)
    if m.group("args"):
        args = ast.literal_eval('(' + m.group('args') + ')')
    else:
        args = []
    return _persisters[m.group("name")](*args)


def PersisterMapper(f):
    persister_path = getattr(f, "_persister", "remoter.InbasePersister()")
    return persister_factory(persister_path)


# -- default persisters --


class Persister(ABC):
    """
    Persists task results and generated metadata in form of a `StoreMsg`

    After calculating the result of a task, a Persister persists the result of the task.
    The metadata how the result is persisted and how it can be obtained again is saved
    in a `StoreMsg`. The StoreMsg is passed on to a store, which distributes it to all
    other stores and saves it.
    """

    path = "abstract"

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        register_persister(cls, cls.path)

    def to_msg(self, task: "Task") -> StoreMsg:
        """Persist task result and conver to StoreMsg"""
        pass

    def from_msg(self, msg: StoreMsg) -> "Task":
        """Load a task result from a storemsg"""
        pass


class InbasePersister(Persister):
    """Persists results as a string in the sqlite task database"""

    path = "remoter.InbasePersister"

    def __init__(self):
        self.path = self.path + "()"

    def to_msg(self, task):
        result = task.result
        msg = StoreMsg(
            task.datetime,
            task.function_module,
            task.function_name,
            task.function_hash,
            task.kwargs,
            result,
            task.task_hash,
            self.path,
            task.failed,
            task.time
        )
        return msg

    def from_msg(self, msg):
        result = msg.result
        return Task(
            self,
            None,
            msg.datetime,
            msg.function_module,
            msg.function_name,
            msg.function_hash,
            msg.kwargs,
            msg.result,
            msg.task_hash,
            True,
            time=msg.time
        )


class FunctionPersister(Persister):
    """Persists results with user defined functions"""

    path = "remoter.FunctionPersister"

    def __init__(self, load_function, save_function, load_kwargs, save_kwargs):
        self.path = self.path + f"('{load_function}', '{save_function}', '{load_kwargs}', '{save_kwargs}')"
        self._load_function = load_function
        self._save_function = save_function
        self._load_kwargs = json.loads(load_kwargs)
        self._save_kwargs = json.loads(save_kwargs)
        self._modules = {}

    def _get_function(self, name):
        module, functionname = name.rsplit(".", 1)
        if module not in self._modules:
            self._modules[module] = importlib.import_module(module)
        f = getattr(self._modules[module], functionname)
        return f

    def _save(self, result):
        return self._get_function(self._save_function)(result, **self._save_kwargs)

    def _load(self, result):
        return self._get_function(self._load_function)(result, **self._load_kwargs)

    def to_msg(self, task):
        result = self._save(task.result)
        msg = StoreMsg(
            task.datetime,
            task.function_module,
            task.function_name,
            task.function_hash,
            task.kwargs,
            result,
            task.task_hash,
            self.path,
            task.failed,
            task.time
        )
        return msg

    def from_msg(self, msg):
        result = self._load(msg.result)
        return Task(
            self,
            None,
            msg.datetime,
            msg.function_module,
            msg.function_name,
            msg.function_hash,
            msg.kwargs,
            result,
            msg.task_hash,
            True,
            msg.time
        )


def save(fn, **kwargs):
    """Indicate which function to use to save the result of this function

    Used by the FunctionPersister
    """
    module = fn.__module__ if fn.__module__ != "__main__" else os.path.split(sys.argv[0])[-1].replace(".py", "")
    n = module + "." + fn.__name__

    def wrapper(f):
        f.__remoter_save = n
        f.__remoter_save_kwargs = json.dumps(kwargs)
        if hasattr(f, "__remoter_load") and hasattr(f, "__remoter_save"):
            l, s = f.__remoter_load, f.__remoter_save
            lkwargs, skwargs = f.__remoter_load_kwargs, f.__remoter_save_kwargs
            f._persister = f"remoter.FunctionPersister('{l}', '{s}', '{lkwargs}', '{skwargs}')"
        return f

    return wrapper


def load(fn, **kwargs):
    """Indicate which function to use to load the result of this function

    Used by the FunctionPersister
    """
    module = fn.__module__ if fn.__module__ != "__main__" else os.path.split(sys.argv[0])[-1].replace(".py", "")
    n = module + "." + fn.__name__

    def wrapper(f):
        f.__remoter_load = n
        f.__remoter_load_kwargs = json.dumps(kwargs)
        if hasattr(f, "__remoter_load") and hasattr(f, "__remoter_save"):
            l, s = f.__remoter_load, f.__remoter_save
            lkwargs, skwargs = f.__remoter_load_kwargs, f.__remoter_save_kwargs
            f._persister = f"remoter.FunctionPersister('{l}', '{s}', '{lkwargs}', '{skwargs}')"
        return f

    return wrapper


class ErrorPersister(Persister):
    """Persists an error as a string in the sqlite task database"""

    path = "remoter.ErrorPersister"

    def __init__(self):
        self.path = self.path + "()"

    def to_msg(self, task):
        msg = StoreMsg(
            task.datetime,
            task.function_module,
            task.function_name,
            task.function_hash,
            task.kwargs,
            error_to_string(task.exception),
            task.task_hash,
            self.path,
            task.failed,
            task.time
        )
        return msg

    def from_msg(self, msg):
        result = msg.result
        return Task(
            self,
            None,
            msg.datetime,
            msg.function_module,
            msg.function_name,
            msg.function_hash,
            msg.kwargs,
            None,
            msg.task_hash,
            True,
            failed=True,
            exception=string_to_error(msg.result),
            time=msg.time
        )

