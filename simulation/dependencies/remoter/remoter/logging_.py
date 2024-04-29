import logging
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union, cast

from .worker import Worker
from .socket_ import PushSocket, Socket

# ==== Logging ====


LOG_RECORDS_ATTRIBUTES = [
    "args",
    "asctime",
    "created",
    "exc_info",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
]


class PushHandler(logging.Handler):
    """A basic logging handler that emits log messages through a Push socket.
    Takes a PUB socket already bound to interfaces or an interface to bind to.
    """

    def __init__(self, interface_or_socket: Union[str, Socket]) -> None:
        logging.Handler.__init__(self)
        self.formatters = {
            logging.DEBUG: logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
            logging.INFO: logging.Formatter("%(message)s\n"),
            logging.WARN: logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
            logging.ERROR: logging.Formatter(
                "%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s - %(exc_info)s\n"
            ),
            logging.CRITICAL: logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s\n"),
        }
        if isinstance(interface_or_socket, PushSocket):
            self.socket = interface_or_socket
        else:
            self.socket = PushSocket()
            self.socket.connect(interface_or_socket)  # type: ignore

    def setFormatter(self, fmt: logging.Formatter, level: int = logging.NOTSET) -> None:
        """Set the Formatter for this handler.
        If no level is provided, the same format is used for all levels. This
        will overwrite all selective formatters set in the object constructor.
        """
        if level == logging.NOTSET:
            for fmt_level in self.formatters.keys():
                self.formatters[fmt_level] = fmt
        else:
            self.formatters[level] = fmt

    def format(self, record: logging.LogRecord) -> str:
        """Format a record."""
        return self.formatters[record.levelno].format(record)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log message on my socket."""
        ei = record.exc_info
        if ei:
            # just to get traceback text into record.exc_text ...
            dummy = self.format(record)
        # See issue #14436: If msg or args are objects, they may not be
        # available on the receiving end. So we convert the msg % args
        # to a string, save it as msg and zap the args.
        d = dict(record.__dict__)
        d["msg"] = record.getMessage()
        d["args"] = None
        d["exc_info"] = None
        # Issue #25685: delete 'message' if present: redundant with 'msg'
        d.pop("message", None)

        self.socket.sync_send_json(d)
        # self.socket.send(bmsg)


