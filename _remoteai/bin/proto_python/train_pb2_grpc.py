import grpc
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities

import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2
import train_pb2 as train__pb2


class TrainServiceStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.ready = channel.unary_unary(
        '/train.train.TrainService/ready',
        request_serializer=train__pb2.default_data.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )
    self.requestAction = channel.unary_unary(
        '/train.train.TrainService/requestAction',
        request_serializer=train__pb2.default_data.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )
    self.onPVPFinish = channel.unary_unary(
        '/train.train.TrainService/onPVPFinish',
        request_serializer=train__pb2.default_data.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )
    self.onPVPStart = channel.unary_unary(
        '/train.train.TrainService/onPVPStart',
        request_serializer=train__pb2.default_data.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )


class TrainServiceServicer(object):

  def ready(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def requestAction(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def onPVPFinish(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def onPVPStart(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_TrainServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'ready': grpc.unary_unary_rpc_method_handler(
          servicer.ready,
          request_deserializer=train__pb2.default_data.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
      'requestAction': grpc.unary_unary_rpc_method_handler(
          servicer.requestAction,
          request_deserializer=train__pb2.default_data.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
      'onPVPFinish': grpc.unary_unary_rpc_method_handler(
          servicer.onPVPFinish,
          request_deserializer=train__pb2.default_data.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
      'onPVPStart': grpc.unary_unary_rpc_method_handler(
          servicer.onPVPStart,
          request_deserializer=train__pb2.default_data.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'train.train.TrainService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class TrainClientStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.startPVP = channel.unary_unary(
        '/train.train.TrainClient/startPVP',
        request_serializer=train__pb2.start_pvp_data.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )
    self.onRequestAction = channel.unary_unary(
        '/train.train.TrainClient/onRequestAction',
        request_serializer=train__pb2.default_data.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )
    self.tick = channel.unary_unary(
        '/train.train.TrainClient/tick',
        request_serializer=train__pb2.train_none.SerializeToString,
        response_deserializer=train__pb2.train_none.FromString,
        )


class TrainClientServicer(object):

  def startPVP(self, request, context):
    """rpc addSample     (default_data)   returns (train_none);
    rpc delSample     (default_data)   returns (train_none);
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def onRequestAction(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def tick(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_TrainClientServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'startPVP': grpc.unary_unary_rpc_method_handler(
          servicer.startPVP,
          request_deserializer=train__pb2.start_pvp_data.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
      'onRequestAction': grpc.unary_unary_rpc_method_handler(
          servicer.onRequestAction,
          request_deserializer=train__pb2.default_data.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
      'tick': grpc.unary_unary_rpc_method_handler(
          servicer.tick,
          request_deserializer=train__pb2.train_none.FromString,
          response_serializer=train__pb2.train_none.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'train.train.TrainClient', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
