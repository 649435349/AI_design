# -*- coding: utf-8 -*-

import asyncore
import socket
import json
import sys
import time
sys.path.append('..')
sys.path.append('../../')

from proto_python import train_pb2
from common.manager import ConnectionManager
from common.manager import SampleOpManager
from common.core import Acceptor
from utils.remoteai_utils import Util
from utils import log4py

logger = log4py.Logger(name='RemoteAIServer')

tick_cnt = 0


class RemoteAIServer(object):
    def __init__(self, service, ip, port):
        self._service = service
        self._service.connection_manager = ConnectionManager.instance()
        self._client_stub_cls = train_pb2.TrainClient_Stub
        self._addr = (ip, port)
        self._acceptor = Acceptor(self._service, self._client_stub_cls)
        self._pvp_id = int(time.time())
        self._last_tick = 0
        logger.debug('RemoteAIServer created.')

    def start(self):
        '''启动服务端，并不断从MQ中读取startPVP请求，发送至训练客户端'''
        self._acceptor.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self._acceptor.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._acceptor.bind(self._addr)
        self._acceptor.listen(5)
        logger.info('RemoteAIServer started.')

        try:
            while True:
                second = (int)(time.time())
                if self._last_tick == 0:
                    self._last_tick = second
                    self.tick()
                elif second - self._last_tick >= 30:
                    self._last_tick = second
                    self.tick()
                Util.load_ops()
                if SampleOpManager.instance().qsize() > 0 and Util.train_client_cnt() > 0:
                    op = SampleOpManager.instance().get_op()
                    if op is not None:
                        if op.has_key('op') and op.has_key('AI') and op.has_key('BT') and op['op'] == 'start' and op['AI'] != '' and op['BT'] != '':
                            self.startPVP(op['AI'], op['BT'])
                asyncore.loop(1, True, None, 1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._acceptor.handle_close()
        logger.info('RemoteAIServer stopped.')

    # --------------------------------
    # 本地接口，服务端 -> 客户端请求，服务端主动
    # --------------------------------
    def startPVP(self, AI, BT, _addr=None):
        '''主动向训练客户端发送startPVP请求，开始一场PVP战斗'''
        data = {'AI': AI, 'BT': BT, "pvpid": self._pvp_id}
        self._pvp_id += 1
        # 不指定具体的客户端，则向所有客户端都发送
        if _addr is None or Util.has_train_client(_addr) == False:
            for stub in Util.all_train_clients():
                stub.startPVP(None, train_pb2.start_pvp_data(
                    data=json.dumps(data)))

            logger.info('RemoteAIServer sent start_pvp request to %d client(s) with data %s.' % (
                Util.train_client_cnt(), data))
        else:
            Util.train_client(_addr).startPVP(
                None, train_pb2.start_pvp_data(data=json.dumps(data)))
            logger.info('RemoteAIServer sent start_pvp request to %s.' % _addr)

    @staticmethod
    def onRequestAction(self, data, _addr=None):
        '''主动向训练客户端发送onRequestAction请求，指导AI进行一下步action'''
        # 不指定具体的客户端，则向所有客户端都发送
        if _addr is None or Util.has_train_client(_addr) == False:
            for stub in Util.all_train_clients():
                stub.onRequestAction(
                    None, train_pb2.default_data(data=json.dumps(data)))

            #logger.info('RemoteAIServer sent on_request_action request to %d client(s) with data %s.' % (Util.train_client_cnt(), data))
        else:
            Util.train_client(_addr).onRequestAction(
                None, train_pb2.default_data(data=json.dumps(data)))
            #logger.info('RemoteAIServer sent on_request_action request to %s.' % _addr)

    def tick(self, _addr=None):
        # 不指定具体的客户端，则向所有客户端都发送
        if _addr is None or Util.has_train_client(_addr) == False:
            for stub in Util.all_train_clients():
                stub.tick(None, train_pb2.train_none())

            logger.info('RemoteAIServer sent tick request to %d client(s).' % Util.train_client_cnt())
        else:
            Util.train_client(_addr).tick(None, train_pb2.train_none())
            logger.info('RemoteAIServer sent tick request to %s.' % _addr)
