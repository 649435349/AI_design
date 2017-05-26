# -*- coding: utf-8 -*-
import grpc, json, sys
sys.path.append('..')
sys.path.append('../../')

from proto_python   import train_pb2
from utils          import log4py

logger = log4py.Logger(name='GrpcClient')

class GrpcClient(train_pb2.TrainClientServicer):
    def __init__(self, owner, ip, port):
        self._owner = owner
        self._channel = grpc.insecure_channel('%s:%d' % (ip, port))
        self._stub = train_pb2.TrainServiceStub(self._channel)
        
    def handle_close(self):
        '''关闭客户端'''
        pass
    
    # 本地接口，客户端 -> 服务端请求，客户端主动
    def ready(self, info):
        request = {'owner': self._owner, 'info': info}
        self._stub.ready(train_pb2.default_data(data=json.dumps(request)))
        logger.info('GrpcClient sent ready request.')
    
    def requestAction(self, info):
        self._stub.requestAction(train_pb2.default_data(data=info))
        logger.info('GrpcClient sent request_action request.')
    
    def onPVPFinish(self, info):
        self._stub.onPVPFinish(train_pb2.default_data(data=info))
        logger.info('GrpcClient sent on_pvp_finish request.')
    
    
    # 远端接口，服务端 -> 客户端请求，客户端被动
    def addSample(self, data, context):
        '''服务端收到客户端的ready消息'''
        logger.info('GrpcClient recieved add_sample request: %s' % data)
        return train_pb2.train_none()
    
    def delSample(self, data, context):
        logger.info('GrpcClient recieved del_sample request: %s' % data)
        return train_pb2.train_none()
    
    def startPvp(self, data, context):
        logger.info('GrpcClient recieved start_pvp request: %s' % data)
        return train_pb2.train_none()
    
