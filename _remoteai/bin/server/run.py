# -*- coding: utf-8 -*-
# python
import asyncore
import threading
import sys
import time
sys.path.append('..')
sys.path.append('../../')

from common.manager import SampleOpManager
from server_service import ServerService
from server import RemoteAIServer
from conf import configs


class SampleOpThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        try:
            while True:
                t = (int)(time.time())
                if t % 10 == 0:
                    op_dict = {'op': 'add', 'sample': t}
                    SampleOpManager.instance().put_op(op_dict)
                    time.sleep(1)
                if SampleOpManager.instance().qsize() >= 10:
                    break
                asyncore.loop(0.2, True, None, 1)
        except KeyboardInterrupt:
            pass
        print 'SampleOpThread exit.'


class ServerThread(threading.Thread):

    def __init__(self, server):
        threading.Thread.__init__(self)
        self._server = server

    def run(self):
        self._server.start()

if __name__ == '__main__':
    server = RemoteAIServer(ServerService(), '0.0.0.0',
                            configs.RPC_SERVER_PORT)
    server.start()
    print 'main exit.'
