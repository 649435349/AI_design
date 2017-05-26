# -*- coding: utf-8 -*-
import threading, json, sys
sys.path.append('..')
sys.path.append('../../')

from proto_python import train_pb2
from client       import RemoteAIClient
from utils        import log4py

logger = log4py.Logger(name='ClientService')

class ClientService(train_pb2.TrainClient):
    def __init__(self, *args, **kwargs):
        super(ClientService, self).__init__(*args, **kwargs)
        self._server_stub = None
        logger.debug('ClientService created.')
    
    def set_server_stub(self, server_stub):
        self._server_stub = server_stub
        
    # --------------------------------
    # 远端接口，服务端 -> 客户端请求，客户端被动处理
    # --------------------------------
    def startPVP(self, b, data, c):
        '''客户端收到startPVP请求，按照服务端发送的样本属性，准备对战双方'''
        logger.info('ClientService recieved start_pvp request: %s' % data)
        if data is None or data.data == '':
            return train_pb2.train_none()
        
        # 异步处理，准备两个对战样本
        request = json.loads(data.data)
        self.onPVPStart(request['AI'], request['BT'])
        return train_pb2.train_none()
    
    def onPVPStart(self, AI, BT):
        '''client_service中收到startPVP请求时，异步调用'''
        info = {'AI': AI, 'BT': BT}
        self._server_stub.onPVPStart(None, train_pb2.default_data(data=json.dumps(info)))
        logger.info('RemoteAIClient sent on_start_pvp request.')
    
    def onRequestAction(self, b, data, c):
        '''客户端收到服务端发送的action结果，调用controller触发机器人动作'''
        logger.info('ClientService reciceved on_request_action: %s' % data)
        return train_pb2.train_none()
    
    def tick(self, b, data, c):
        #logger.info('RemoteAIClient recieved tick request')
        return train_pb2.train_none()
    