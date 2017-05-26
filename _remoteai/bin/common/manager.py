# -*- coding: utf-8 -*-

import Queue, time, sys
sys.path.append('..')
sys.path.append('../../')

from singleton import Singleton
from utils     import log4py

logger = log4py.Logger(name='common')

class ConnectionManager(Singleton):
    def __init__(self):
        self._connect_map = {}
        self._connect_host_map = {}

    @property
    def timestamp(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def add_connection(self, connection):
        if self._connect_map.get(connection.addr):
            logger.error('One ip addr only one connection.')
            return
        self._connect_map[connection.addr] = connection
        logger.debug('ConnectionManager add_connection: %s' % connection)

    def del_connection(self, connection):
        self._connect_map.pop(connection.addr, None)
        logger.debug('ConnectionManager del_connection: %s' % connection)

    def clr_connection(self):
        for conn in self._connect_map.values():
            conn.handle_close()
        logger.debug('ConnectionManager clr_connection.')
    
    def get_connection(self, addr):
        logger.debug('ConnectionManager get_connection.')
        return self._connect_map.get(addr)
    
    def add_host(self, addr, host_id):
        self._connect_host_map[host_id] = addr
        logger.debug('ConnectionManager add_host.')
    
    def get_addr_by_host_id(self, host_id):
        logger.debug('ConnectionManager get_addr_by_host_id.')
        return self._connect_host_map.get(host_id)
    
    def get_all_connections(self):
        for addr in self._connect_map.keys():
            yield self._connect_map[addr]
    
    def print_out(self):
        for addr in self._connect_map.keys():
            print '%s, %s' % (addr, self._connect_map[addr])

class ClientManager(Singleton):
    def __init__(self):
        self._client_stub_map = {}
        self._train_client_stub_map = {}
        logger.debug('ClientManager created.')
    
    def add_train_client_stub(self, addr, client_stub):
        self._train_client_stub_map[addr] = client_stub
        logger.debug('ClientManager add_train_client_stub.')
        
    def add_client_stub(self, addr, client_stub):
        self._client_stub_map[addr] = client_stub
        logger.debug('ClientManager add_client_stub.')
    
    def get_train_client_stub(self, addr):
        logger.debug('ClientManager get_train_client_stub.')
        return self._train_client_stub_map.get(addr)
    
    def get_client_stub(self, addr):
        logger.debug('ClientManager get_client_stub.')
        return self._client_stub_map.get(addr)
      
    def del_client_stub(self, addr):
        self._client_stub_map.pop(addr, None)
        self._train_client_stub_map.pop(addr, None)
        logger.debug('ClientManager del_client_stub.')

    def clr_client_stub(self):
        self._client_stub_map.clear()
        self._train_client_stub_map.clear()
        logger.debug('ClientManager clr_client_stub.')
    
    def get_all_train_addrs(self):
        return self._train_client_stub_map.keys()
    
    def get_all_addrs(self):
        return self._client_stub_map.keys()
    
    def get_all_train_client_stubs(self):
        for addr in self._train_client_stub_map.keys():
            yield self._train_client_stub_map[addr]
            
    def get_all_client_stubs(self):
        for addr in self._client_stub_map.keys():
            yield self._client_stub_map[addr]
    
    def has_train_client_stub(self, addr):
        return self._train_client_stub_map.has_key(addr)
       
    def has_client_stub(self, addr):
        return self._client_stub_map.has_key(addr)
    
    def print_out(self):
        for addr in self._client_stub_map.keys():
            print '%s, %s' % (addr, self._client_stub_map[addr])
    
class SampleOpManager(Singleton):
    def __init__(self):
        self._queue = Queue.Queue()
    
    def put_op(self, op_dict):
        self._queue.put(op_dict)
    
    def get_op(self):
        if self._queue.empty() == False:
            return self._queue.get()
        else:
            return None
    
    def full(self):
        return self._queue.full()
    
    def empty(self):
        return self._queue.empty()
    
    def qsize(self):
        return self._queue.qsize()
    