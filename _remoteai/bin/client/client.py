# -*- coding: utf-8 -*-
import asyncore, json, socket, sys, time
sys.path.append('..')
sys.path.append('../../')

from common.core  import RpcChannel
from proto_python import train_pb2
from utils        import log4py

logger = log4py.Logger(name='RemoteAIClient')

class RemoteAIClient(object):
	def __init__(self, owner, service, ip, port):
		self._owner = owner
		self._service = service
		self._channel = RpcChannel(self._service)
		self._server_stub = train_pb2.TrainService_Stub(self._channel)
		self._channel.connect((ip, port))
		self._service.set_server_stub(self._server_stub)
		self._connected = True
		logger.info('RemoteAIClient connect to %s:%d success.' % (ip, port))
		
	@property
	def isConnected(self):
		logger.debug('RemoteAIClient connected.')
		return self.connected
	
	def handle_close(self):
		self._connected = False
		logger.debug('RemoteAIClient closed.')
	
	@property
	def server(self):
		return self._server_stub
	
	def start(self):
		ready_request = {'ip': socket.gethostbyname(socket.gethostname()), 'name': socket.gethostname()}
		self.ready(ready_request)
		try:
			while True:
				t = (int)(time.time())
				if t%10 == 3:
					self.requestAction('action-%d' % t)
					time.sleep(1)
				if t%10 == 7:
					self.onPVPFinish('finish-%d' % t)
					time.sleep(1)
				asyncore.loop(0.2, True, None, 1)
		except KeyboardInterrupt:
			self.handle_close()
	
	# --------------------------------
	# 本地接口，客户端 -> 服务端请求，客户端主动
	# --------------------------------
	def ready(self, info):
		'''启动时调用，通知服务端已准备就绪'''
		request = {'client_type': self._owner, 'info': info}
		self.server.ready(None, train_pb2.default_data(data=json.dumps(request)))
		logger.info('RemoteAIClient sent ready request.')
	
	def requestAction(self, info):
		'''run中定时调用，每个tick请求一次action'''
		self.server.requestAction(None, train_pb2.default_data(data=info))
		logger.info('RemoteAIClient sent request_action request.')
	
	def onPVPFinish(self, info):
		'''战斗分出胜负时调用'''
		self.server.onPVPFinish(None, train_pb2.default_data(data=info))
		logger.info('RemoteAIClient sent on_pvp_finish request.')
	
		
