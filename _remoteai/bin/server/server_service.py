# -*- coding: utf-8 -*-
import json
import threading
import sys
import random

sys.path.append('..')
sys.path.append('../../')

from proto_python import train_pb2
from utils.remoteai_utils import Util
from utils import log4py
from common.manager import ClientManager
from conf import configs
from server import RemoteAIServer
#from algo import ai_dqn_fyf,ai_dueling_dqn_fyf,ai_static_fyf
from algo import ai_trpo
# from algo import ai_pg


logger = log4py.Logger(name='ServerService')


class ServerService(train_pb2.TrainService):

    def __init__(self, *args, **kwargs):
        super(ServerService, self).__init__(*args, **kwargs)
        #self.agent = ai_dqn_fyf.mc_agent_dqn_fyf(resume=False, who_vs_who='61')
        #self.agent = ai_dueling_dqn_fyf.mc_agent_dueling_dqn_fyf(resume=True,who_vs_who='61')
        #self.agent=ai_static_fyf.mc_agent_static_fyf(who_vs_who='11')
        self.agent=ai_trpo.mc_trpo(resume=False,who_vs_who='11')
        logger.debug('ServerService created.')

    @classmethod
    def create_service(cls):
        return train_pb2.beta_create_TrainService_server(cls())

    # --------------------------------
    # 远端接口，客户端 -> 服务端请求，服务端被动处理
    # --------------------------------
    def ready(self, b, data, c):
        '''收到客户端的ready信息，根据客户端类型，添加到ClientManager中，用于选择服务端主动发送消息的标的'''
        logger.info('ServerService recieved ready request: %s' % data)
        if data is not None and data.data != '' and Util.train_client_cnt() == 0:
            request = json.loads(data.data)

            # 写入pvp数据到MQ，开始第一批对局
            self.agent.startPVP()

            # 区分训练客户端和正式客户端，暂时不处理，上线后区分
            # if request.has_key('client_type') and request['client_type'] ==
            # configs.TRAIN_CLIENT_TYPE:
            ClientManager.instance().add_train_client_stub(
                self.rpc_channel.addr, self.rpc_channel._client_stub_cls)
        return train_pb2.train_none()

    def onPVPStart(self, b, data, c):
        '''收到客户端响应startPVP请求，由客户端传入准备好的对战双方的信息，本地调用算法模块处理'''
        logger.info('ServerService recieved on_start_pvp request: %s' % data)
        if data is None or data.data == '':
            return train_pb2.train_none()
        #data.data example:{u'self': {u'refuseSkillids': {u'1050': u'no target', u'1040': u'no target', u'1150': u'no target', u'1100': u'no target', u'1080': u'no target'}, u'hp': 2007, u'skillids': [1060, 1170, 1070, 1160, 1110], u'id': 29100, u'position': u'(5096.062500, 1661.153809, 5238.458008)'}, u'other': {}, u'pvpid': 1491983132, u'name': u'trainer6075'}
        # TODO: 调用算法模块algoai.start方法，传入request，返回值为一个动作

        start_pvp_data = json.loads(data.data)
        action_data=self.agent.add_start_data(start_pvp_data)

        RemoteAIServer.onRequestAction(RemoteAIServer, data=action_data)
        return train_pb2.train_none()

    def requestAction(self, b, data, c):
        '''收到客户端的requestAction请求，调用算法模块返回结果'''
        # logger.info('ServerService recieved request_action request: %s' % data)
        if data is None or data.data == '':
            return train_pb2.train_none()

        # TODO:调用算法模块algoai.action方法，传入参数request，在该方法中调用server.onRequestAction将处理结果返回到客户端
        state = json.loads(data.data)
        action_data = self.agent.get_action(state)
        #action_data = {"trainer_name": state['name'], "step_id": 1, "action_parameters": {"skill_id": random.choice(state["self"]["skillids"])}, "action_id": 1}

        RemoteAIServer.onRequestAction(RemoteAIServer, data=action_data)
        return train_pb2.train_none()

    def onPVPFinish(self, b, data, c):
        '''收到客户端的onPVPFinish请求，异步记录战斗结果'''
        #logger.info('ServerService recieved on_pvp_finish request: %s' % data)
        if data is None or data.data == '':
            return train_pb2.train_none()

        # TODO: 调用算法模块algoai.finish方法，传入参数request，返回值为空
        finish_pvp_data = json.loads(data.data)
        self.agent.add_finish_data(finish_pvp_data)
        # data = algoai.finish(request)
        return train_pb2.train_none()


