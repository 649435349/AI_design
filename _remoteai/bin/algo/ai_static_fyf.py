# -*- coding: utf-8 -*-

import numpy as np
import random
import json
import threading
from conf import configs
from utils import log4py
import cPickle as pickle
import copy
from collections import defaultdict,deque

from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.initializers import glorot_normal

logger = log4py.Logger(name='AI train', log_file=configs.DEFAULT_LOGGER_FILE)


# BT使用的技能列表
skill_ids={'1':[1060,1100,1070,1150,1000],'6':[3120, 3060, 3050, 3080, 3000],'5':[2210, 2280, 2310, 2120, 2290, 2080]}
hps={'1':2007,'6':987,'5':987}

class mc_agent_static_fyf():
    '''
    直接按照既定的顺序放吧
    '''
    def __init__(self,who_vs_who):
        # 对阵的职业，默认为战士和战士
        self.who_vs_who=who_vs_who

        # 训练的次数
        self.batch_size = 5
        self.iter_num = 200
        self.iter = 0

        # 初始化整体参数
        self.reset_par()

        # 远离和靠近的距离
        self.move_actions = [-30, 30]

        # reward
        self.win_cnt = 0
        self.lose_cnt = 0


        self.wpct_file = open(configs.WPCT_FILE, 'wb+')
        #self.model_file = open('../../model/'+self.who_vs_who, 'wb+')

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0
        # 为finish_cnt上锁，防止同时访问
        self.finish_cnt_lock = threading.Lock()


    def startPVP(self, MQ_file=configs.DEFAULT_MQ_FILE):
        """
        将PVP双方信息写入MQ，开始一批PVP对局
        """
        self.reset_par()  # 首先重置参数
        role_data=json.dumps({'op':'start',"AI": {"career": int(self.who_vs_who[0]), "level": 35}, "BT": {"career": int(self.who_vs_who[1]), "level": 35}})
        with open(MQ_file, 'w') as f:
            for i in range(self.batch_size):
                f.write(role_data + '\n')

    def add_start_data(self, data):
        """
        获取onPVPStart时传入的数据
        """
        trainer = data['name']
        self.data[trainer]['step_id'] = 0
        # 统计开始的pvp对局数
        self.start_cnt += 1
        can_action = self.__format_state(data)
        self.data[trainer]['step_id'] += 1
        action_ind = self.get_epsi_greedy_action(can_action)
        logger.info('trainer:{}, step_id:{}, action_id:{}'.format(
            trainer, self.data[trainer]['step_id'], action_ind))

        return self.__format_action(action_ind, trainer)

    def __format_action(self, action_ind, trainer):
        """
        将action编号转化为对应的格式化action数据
        action_id:1为放技能,2为移动.
        """
        if action_ind < len(skill_ids[self.who_vs_who[0]]):
            action_data = {"trainer_name": trainer, "step_id": self.data[trainer][
                'step_id'], "action_parameters": {"skill_id": skill_ids[self.who_vs_who[0]][action_ind]}, "action_id": 1}
        elif action_ind < len(skill_ids[self.who_vs_who[0]]) + len(self.move_actions):
            action_ind = action_ind - len(skill_ids[self.who_vs_who[0]])
            action_data = {"trainer_name": trainer, "step_id": self.data[trainer][
                'step_id'], "action_parameters": {"move_distance": self.move_actions[action_ind]}, "action_id": 2}
        else:
            raise ValueError(
                'action_ind:{} is out of range!'.format(action_ind))
        return action_data

    def __format_state(self, state):
        """
        从request_action传入的原始data中提取需要的state数据和当前可放的技能的列表
        """
        # 获取可用技能列表，so far设置为可用，普攻一直可用
        skill_available = state['self']['skillids']
        skill_state = [0 for i in range(len(skill_ids[self.who_vs_who[0]]))]
        for (i, skill) in enumerate(skill_ids[self.who_vs_who[0]]):
            if skill in skill_available:
                skill_state[i] = 1
            if skill in state['self']['refuseSkillids'] and (state['self']['refuseSkillids'][skill] == 'so far'):
                skill_state[i] = 2
            if i == len(skill_ids[self.who_vs_who[0]]) - 1:
                skill_state[i] = 1
        return skill_state

    def get_action(self, state):
        """
        根据环境状态信息输出action
        """
        trainer = state['name']
        if trainer not in self.data:
            return

        can_action= self.__format_state(state)

        self.data[trainer]['step_id'] += 1

        action_ind = self.get_epsi_greedy_action(can_action)
        logger.info('trainer:{}, step_id:{}, action_id:{}'.format(
            trainer, self.data[trainer]['step_id'], action_ind))

        return self.__format_action(action_ind, trainer)

    def get_epsi_greedy_action(self,can_action):
        for i in [4,0,1,3,2]:
            if can_action[i]==1:
                return i
        return 5

    def add_finish_data(self, data):
        """
        获取onPVPFinish时传入的数据
        """
        trainer = data['name']

        if data['isWin']:
            self.win_cnt+=1
        else:
            self.lose_cnt+=1
        # 统计完成的pvp对局数，先获取互斥锁
        with self.finish_cnt_lock:
            self.finish_cnt += 1
        logger.info('trainer:{}, finish_info:{}'.format(
            trainer, str(data)))

        log_info = 'iter:{}, win_ratio:{}, trainer:{},  win_cnt:{}, lose_cnt:{} \n'.format(
            self.iter, 1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt), trainer, self.win_cnt, self.lose_cnt)

        logger.info(log_info)
        self.wpct_file.writelines(','.join([str(self.iter), str(1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt)), '\n']))

        # 如果开始的pvp都完成了
        if self.finish_cnt >= self.batch_size:
            self.wpct_file.flush()
            self.iter += 1
            if self.iter < self.iter_num:
                self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))