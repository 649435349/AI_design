# -*- coding: utf-8 -*-
# python
import sys
sys.path.append('..')
sys.path.append('../../')

from client_service import ClientService
from client         import RemoteAIClient
from conf           import configs

if __name__ == '__main__':
	client = RemoteAIClient(configs.TRAIN_CLIENT_TYPE, ClientService(), configs.RPC_SERVER_HOST, configs.RPC_SERVER_PORT)
	client.start()