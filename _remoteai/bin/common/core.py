#-*- coding: utf-8 -*-

import asyncore, socket, struct, StringIO, sys, weakref
sys.path.append('..')
sys.path.append('../../')

from proto_python    import common_pb2
from proto_python    import train_pb2
from google.protobuf import service
from manager         import ConnectionManager
from manager         import ClientManager
from utils           import log4py


logger = log4py.Logger(name='common')

class RpcController(service.RpcController):
    def __init__(self, channel):
        self.rpc_channel = channel

class Connection(asyncore.dispatcher):
    RECV_BUFFER_SIZE = 1024 * 1024
    def __init__(self, channel, sock=None):
        asyncore.dispatcher.__init__(self, sock)
        ConnectionManager.instance().add_connection(self)
        self._channel = channel
        self._w_buffer = StringIO.StringIO()
        self._r_buffer = ""
        if self.socket is None:
            self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setblocking(0)
        logger.debug('Connection created.')

    def handle_close(self):
        self.close()
        ConnectionManager.instance().del_connection(self)
        ClientManager.instance().del_client_stub(self.addr)
        logger.debug('Connection closed.')

    def handle_read(self):
        data = self.recv(self.RECV_BUFFER_SIZE)
        self._r_buffer += data
        self._channel.on_request()
        logger.debug('Connection handle_read.')

    def writable(self):
        #logger.debug('Connection writeable.')
        return self._w_buffer.getvalue()

    def handle_write(self):
        buff = self._w_buffer.getvalue()
        if buff:
            sent = self.send(buff)
            self._w_buffer = StringIO.StringIO(buff[sent:])
            self._w_buffer.seek(0, 2)
        logger.debug('Connection handle_write.')

    def send_data(self, data):
        self._w_buffer.write(data)
        logger.debug('Connection send_data.')

    def read_data(self):
        logger.debug('Connection read_data.')
        return self._r_buffer

    def consume_data(self, count):
        if count:
            self._r_buffer = self._r_buffer[count:]
        logger.debug('Connection consume_data.')

class RpcChannel(service.RpcChannel):
    def __init__(self, service, sock=None):
        super(RpcChannel, self).__init__()
        self._call_index = 0
        self._service = service
        self._connection = Connection(self, sock)
        self._callbacks = {}
        self._service.rpc_channel = weakref.proxy(self)
        logger.debug('RpcChannel created.')
    
    @property
    def addr(self):
        return self._connection.addr
        
    def get_call_index(self):
        self._call_index += 1
        logger.debug('RpcChannel get_call_index.')
        return self._call_index

    def connect(self, addr):
        self._connection.connect(addr)
        logger.debug('RpcChannel connect.')

    def on_request(self):
        buff = self._connection.read_data()
        def read_data(size):
            data = buff[self.consume:self.consume+size]
            self.consume += size
            return data

        def enough_data(size):
            return len(buff) - self.consume >= size

        def unpack(fmt, size):
            data = read_data(size)
            return struct.unpack(fmt, data)
        
        logger.debug('RpcChannel on_request.')
        total_consume = self.consume = 0
        while True:
            if not enough_data(4):
                break
            total_len = unpack("<I", 4)[0]

            if not enough_data(total_len):
                break

            name_len = unpack("<I", 4)[0]
            msg_name = read_data(name_len)
            msg_data = read_data(total_len - name_len)

            msg_type = getattr(common_pb2, msg_name)
            msg = msg_type()
            msg.ParseFromString(msg_data)

            if msg_name == "Request":
                self.handle_request(msg)
            elif msg_name == "Response":
                self.handle_response(msg)

            total_consume = self.consume
            
        logger.debug('RpcChannel on_request consume: %d, %s' % (total_consume, repr(self._connection._r_buffer)))
        self._connection.consume_data(total_consume)

    def CallMethod(self, method_descriptor, rpc_controller, request, response_class, done):
        cmd_index = method_descriptor.index
        assert cmd_index < 65536

        index = self.get_call_index()
        self._send_request(index, method_descriptor, request, done)
        logger.debug('RpcChannel CallMethod.')

    def handle_request(self, msg):
        method = self._service.GetDescriptor().FindMethodByName(msg.remote_call.rpc_name)
        request = self._service.GetRequestClass(method)()
        logger.debug('RpcChannel handle_request: %s, %s' % (type(request), repr(msg.remote_call.msg.data)))
        request.ParseFromString(msg.remote_call.msg.data)
        response = self._service.CallMethod(method, None, request, self._connection.addr)
        self._send_response(msg.index, response)

    def handle_response(self, msg):
        if msg.msg.name:
            message_type = getattr(train_pb2, msg.msg.name)
            message = message_type()
            message.ParseFromString(msg.msg.data)
        else:
            message = None
            
        logger.debug('RpcChannel handle_response.')
        cb = self._callbacks.pop(msg.index, None)
        if cb:
            cb(message)

    def _send_request(self, index, method, request, callback=None):
        msg = common_pb2.Request(
                index=index,
                remote_call=common_pb2.Rpc(
                    rpc_name=method.name,
                    msg=common_pb2.Message(
                        name=request.DESCRIPTOR.name,
                        data=request.SerializeToString()
                        )
                    )
                )

        msg_name = msg.DESCRIPTOR.name
        msg_data = msg.SerializeToString()

        name_len = len(msg_name)
        total_len = name_len + len(msg_data)
        self._connection.send_data(struct.pack("<II", total_len, name_len))
        self._connection.send_data(msg_name)
        self._connection.send_data(msg_data)

        self._callbacks[index] = callback
        logger.debug('RpcChannel _send_request.')

    def _send_response(self, index, message):
        msg = common_pb2.Response(
                index=index,
                msg=common_pb2.Message(
                    name=message.DESCRIPTOR.name,
                    data=message.SerializeToString()
                    )
                )

        msg_name = msg.DESCRIPTOR.name
        msg_data = msg.SerializeToString()

        name_len = len(msg_name)
        total_len = name_len + len(msg_data)
        self._connection.send_data(struct.pack("<II", total_len, name_len))
        self._connection.send_data(msg_name)
        self._connection.send_data(msg_data)
        logger.debug('RpcChannel _send_response.')

class Acceptor(asyncore.dispatcher):
    def __init__(self, service, client_stub_cls):
        asyncore.dispatcher.__init__(self)
        self._service = service
        self._client_stub_cls = client_stub_cls
        logger.debug('Acceptor created.')

    def handle_accept(self):
        ret = self.accept()
        if ret is not None:
            sock, addr = ret
            channel = RpcChannel(self._service, sock)
            channel._client_stub_cls = self._client_stub_cls(channel)
            ClientManager.instance().add_client_stub(addr, channel._client_stub_cls)
        logger.debug('Acceptor handle_accept.')

    def handle_close(self):
        self.close()
        ConnectionManager.instance().clr_connection()
        ClientManager.instance().clr_client_stub()
        logger.debug('Acceptor handle_close.')
