# -*- coding: utf-8 -*-
import json, sys
sys.path.append('..')
sys.path.append('../../')

from common.manager import ClientManager
from common.manager import SampleOpManager
from conf import configs

class Util(object):
    @staticmethod
    def has_client(addr):
        return ClientManager.instance().has_client_stub(addr)
    
    @staticmethod
    def has_train_client(addr):
        return ClientManager.instance().has_train_client_stub(addr)
    
    @staticmethod
    def client(addr):
        return ClientManager.instance().get_client_stub(addr)
    
    @staticmethod
    def train_client(addr):
        return ClientManager.instance().get_train_client_stub(addr)
    
    @staticmethod
    def client_cnt():
        return len(ClientManager.instance().get_all_addrs())
    
    @staticmethod
    def train_client_cnt():
        return len(ClientManager.instance().get_all_train_addrs())
    
    @staticmethod
    def all_clients():
        return ClientManager.instance().get_all_client_stubs()
    
    @staticmethod
    def all_train_clients():
        return ClientManager.instance().get_all_train_client_stubs()
    
    @staticmethod
    def load_ops(file_path=configs.DEFAULT_MQ_FILE):
        # read PVPs from MQ
        # {"op": "start", "trainer": {"level": 50, "attack": 100, "skills": [1,2,3]}, "opponent": {"level": 50, "attack": 100, "skills": [1,2,3]}}
        i = 0
        input_file = open(file_path, 'r')
        for line in input_file:
            if line is None or line == '':
                continue
            try:
                SampleOpManager.instance().put_op(json.loads(line))
            except Exception:
                pass
            i += 1
        input_file.close()
        
        # clean MQ
        if i > 0:
            output_file = open(file_path, 'w')
            output_file.write('')
            output_file.close()
    