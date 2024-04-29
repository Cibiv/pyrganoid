from typing import Optional
from types import TracebackType

import pickle
import json

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

def error_to_string(err: Optional[tuple[BaseException, TracebackType]]) -> Optional[str]:
    if err:
        return str(err)
        #return pickle.dumps(err, protocol=pickle.HIGHEST_PROTOCOL).decode("utf8", "surrogateescape")
    return None

def string_to_error(s: Optional[str]) -> Optional[tuple[BaseException, TracebackType]]:
    if s:
        return s, None
        return pickle.loads(s.encode("utf8", "surrogateescape"))
    return None


def serialize_value(value):
    from .task import Task
    if isinstance(value, Task):
        value.calc_hash()
        return ('t', value.task_hash)
    if isinstance(value, list):
        return ('l', [serialize_value(v) for v in value])
    if isinstance(value, dict):
        return ('d', {key: serialize_value(v) for key, v in value.items()})
    else:
        return ('a', value)

def deserialize_value(value, store):
    value_type, value = value
    if value_type == 't':
        task = store.load_from_hash(value)
        if task is None:
            raise Exception(f'Could not deserialize task with hash {value["value"]}')
        return task
    elif value_type == 'l':
        return [deserialize_value(v, store) for v in value]
    elif value_type == 'd':
        return {key: deserialize_value(v, store) for key, v in value.items()}
    elif value_type == 'a':
        return value

def serialize_kwargs(kwargs):
    if kwargs is not None:
        kwargs = {key: serialize_value(value) for key, value in kwargs.items()}
        kwargs = _clean(kwargs)
    return json.dumps(kwargs)

def deserialize_kwargs(string, store):
    kwargs = json.loads(string)
    if kwargs is not None:
        kwargs = {key: deserialize_value(value, store) for key, value in kwargs.items()}
    return kwargs

def _clean(v):
    if isinstance(v, dict):
        r = {}
        for key in sorted(v):
            r[key] = _clean(v[key])
        return r
    return v

