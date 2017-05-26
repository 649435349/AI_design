# -*- coding: utf-8 -*-

import grpc, sys
sys.path.append('..')

from concurrent   import futures
from proto_python import common_pb2
from proto_python import train_pb2
from utils        import log4py

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
logger = log4py.Logger(name='GrpcServer')

class GrpcServer(train_pb2.TrainServiceServicer):
    def __init__(self, ip, port):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        train_pb2.add_TrainServiceServicer_to_server(self, self._server)
        self._stub = train_pb2.TrainClient_Stub
        self._addr = (ip, port)
        self._server.add_insecure_port('[::]:%d' % port)
        self._client_ready = False
    
    def start(self):
        self._server.start()
        logger.info('GrpcServer started');
    
    def stop(self):
        self._server.stop(0)
        logger.info('GrpcServer stoped');
        
    # 本地接口，客户端 -> 服务端请求，客户端主动
    def addSample(self, info):
        self._stub.ready(train_pb2.default_data(data=info))
        logger.info('GrpcServer sent add_sample request.')
    
    def delSample(self, info):
        self._stub.requestAction(train_pb2.default_data(data=info))
        logger.info('GrpcServer sent del_sample request.')
    
    def startPvp(self, info):
        self._stub.onPVPFinish(train_pb2.default_data(data=info))
        logger.info('GrpcServer sent start_pvp request.')
    
    
    # 远端接口，客户端 -> 服务端请求，服务端被动
    def ready(self, data, context):
        '''服务端收到客户端的ready消息'''
        logger.info('GrpcServer recieved ready request: %s' % data)
        self._client_ready = True
        return train_pb2.train_none()
    
    def requestAction(self, data, context):
        logger.info('GrpcServer recieved request_action request: %s' % data)
        return train_pb2.train_none()
    
    def onPVPFinish(self, data, context):
        logger.info('GrpcServer recieved on_pvp_finish request: %s' % data)
        return train_pb2.train_none()
    
