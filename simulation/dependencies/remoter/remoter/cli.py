import click
import logging

from .worker import Worker
from .server import Server

@click.group()
@click.version_option(version='0.1')
@click.option("--debuglevel", default="INFO")
def remoter(debuglevel):
    logging.basicConfig(format="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s", level=debuglevel)

@remoter.command()
@click.argument('address', default="ipc://.remoter.ipc")
@click.option("--logfile", default=None)
def worker(address, logfile):
    if logfile:
        import logging
        root_logger = logging.getLogger()
        stream_handler = logging.FileHandler(logfile)
        root_logger.addHandler(stream_handler)
    worker = Worker(address)
    worker.start_async()

@remoter.command()
@click.argument('address', default="ipc://.remoter.ipc")
def server(address):
    server = Server(address)
    server.start_async()

if __name__ == "__main__":
    remoter()
