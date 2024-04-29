import json
import zmq
import trio

class Socket:

    _waiting_time = 0.001

    def __init__(self, ctx=None, **kwargs):
        if ctx is None:
            ctx = zmq.Context.instance()
        self.ctx = ctx
        self._socket = self._make_socket()

    def connect(self, addr):
        return self._socket.connect(addr)

    def listen(self, addr):
        return self._socket.bind(addr)

    def recv(self):
        msg = self._socket.recv()
        return msg.decode("utf8")

    async def async_recv(self):
        while True:
            try:
                msg = self._socket.recv(zmq.DONTWAIT)
                return msg.decode("utf8")
            except zmq.error.Again:
                await trio.sleep(self._waiting_time)

    def recv_json(self):
        msg = self.recv()
        return json.loads(msg)

    async def async_recv_json(self):
        msg = await self.async_recv()
        return json.loads(msg)

    def send(self, msg):
        return self._socket.send(msg.encode("utf8"))

    async def async_send(self, msg):
        return self.send(msg)

    def send_json(self, msg):
        msg_serialized = json.dumps(msg)
        r = self.send(msg_serialized)
        return r

    async def async_send_json(self, msg):
        self.send_json(msg)

    def close(self):
        return self._socket.close()


class PushSocket(Socket):
    def _make_socket(self):
        return self.ctx.socket(zmq.PUSH)


class PullSocket(Socket):
    def _make_socket(self):
        return self.ctx.socket(zmq.PULL)


class ReqSocket(Socket):
    def _make_socket(self):
        return self.ctx.socket(zmq.REQ)


class RepSocket(Socket):
    def _make_socket(self):
        return self.ctx.socket(zmq.REP)
