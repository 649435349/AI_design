# -*- coding: utf-8 -*-

import numpy as np
import random
import json
import threading
from conf import configs
from utils import log4py
from .trpo_agent import agent
from .trpo_agent.core import PG_OPTIONS
from .trpo_agent.misc_utils import *
import cPickle as pickle
import copy
from collections import defaultdict,deque


logger = log4py.Logger(name='AI train', log_file=configs.DEFAULT_LOGGER_FILE)
cfg = update_default_config(PG_OPTIONS)

# BT使用的技能列表
skill_ids={'1':[1100, 1150, 1070, 1060, 1170, 1000],'6':[3080,3060 ,3070, 3140, 3100, 3000],'5':[2210, 2280, 2310, 2120, 2290, 2080]}
hps={'1':2007,'6':987,'5':987}


def get_distance(pos1, pos2):
    """计算两点之间距离"""
    pos1 = np.array(eval(pos1))
    pos2 = np.array(eval(pos2))
    return int(np.linalg.norm(pos1 - pos2))

class mc_trpo():
    def __init__(self, resume,who_vs_who):
        # 是否用现有的训练好的网络
        self.resume = resume

        # 对阵的职业，默认为战士和战士
        self.who_vs_who=who_vs_who

        # 训练的次数
        self.batch_size = 5
        self.iter_num = 200
        self.iter = 0

        # 远离和靠近的距离
        self.move_actions = [-30, 30]

        # 初始化整体参数
        self.reset_par()

        # reward
        self.win_cnt = 0
        self.lose_cnt = 0

        # trpo_agent
        self.agent=agent.TrpoAgent()

        self.paths=[]

        self.win_memory=deque([],1000)
        self.lose_memory = deque([], 500)

        self.wpct_file = open(configs.WPCT_FILE, 'wb+')

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0
        # 为finish_cnt上锁，防止同时访问
        self.finish_cnt_lock = threading.Lock()

        self.paths=[]

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
        # 用于存储对局中间数据
        self.data[trainer]['states'] = []
        self.data[trainer]['actions'] = []
        self.data[trainer]['rewards'] = []
        self.data[trainer]['prob'] = []
        self.data[trainer]['initial_info'] = {'hp_self': data['self']['hp'],
                                              'hp_other': data['other']['hp'] if data['other'] else hps[self.who_vs_who[1]],
                                              'dist': get_distance(data['self']['position'],
                                                                   data['other']['position']) if data[
                                                  'other'] else 500
                                              }
        can_action, now_state = self.__format_state(data)
        self.data[trainer]['states'].append(now_state)

        self.data[trainer]['step_id'] += 1

        action_ind = self.get_epsi_greedy_action(trainer,can_action, now_state)

        self.data[trainer]['actions'].append(action_ind)
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
                skill_state[i] = 1
            if i == len(skill_ids[self.who_vs_who[0]]) - 1:
                skill_state[i] = 1

        trainer = state['name']
        hp_self = 1.0 * state['self']['hp'] / \
                  self.data[trainer]['initial_info']['hp_self']
        try:
            hp_other = 1.0 * state['other']['hp'] / \
                       self.data[trainer]['initial_info']['hp_other']
        except:
            hp_other=1.0
        try:
            dist =get_distance(state['self']['position'], state['other']['position'])
        except:
            dist=500
        state_vec = np.array([hp_self, hp_other] +[dist]+skill_state)
        return skill_state,state_vec

    def get_action(self, state):
        """
        根据环境状态信息输出action
        """
        trainer = state['name']
        if trainer not in self.data:
            return

        can_action,now_state = self.__format_state(state)
        self.data[trainer]['states'].append(now_state)

        self.data[trainer]['step_id'] += 1

        action_ind = self.get_epsi_greedy_action(trainer,can_action,now_state)

        self.data[trainer]['actions'].append(action_ind)
        logger.info('trainer:{}, step_id:{}, action_id:{}'.format(
            trainer, self.data[trainer]['step_id'], action_ind))

        return self.__format_action(action_ind, trainer)

    def get_epsi_greedy_action(self,trainer,can_action,now_state):
        action, agentinfo = self.agent.act(now_state)
        print action
        self.data[trainer]['prob'].append(agentinfo['prob'])
        return action


    def get_memoryset(self,trainer):
        '''
        构造(state,action,reward,next_state)对
        '''
        res=[]
        len1,len2,len3=len(self.data[trainer]['states']),len(self.data[trainer]['actions']),len(self.data[trainer]['rewards'])
        if np.var([len1,len2,len3]):
            print 'lens are not equal,please check.'

        for i in range(len1-1):
            res.append((self.data[trainer]['states'][i],self.data[trainer]['actions'][i],
                        self.data[trainer]['rewards'][i],self.data[trainer]['states'][i+1]))
        #res.append((self.data[trainer]['states'][len1-1], self.data[trainer]['actions'][len1-1],
        #            self.data[trainer]['rewards'][len1-1],(self.data[trainer]['finish_info']['hp_self'],self.data[trainer]['finish_info']['hp_other'])))
        return res

    def add_finish_data(self, data):
        """
        获取onPVPFinish时传入的数据
        """
        trainer = data['name']
        if trainer not in self.data:
            return
        self.data[trainer]['finish_info'] = {'hp_self': 1.0*data['self']['hp']/self.data[trainer]['initial_info']['hp_self'],
                                             'hp_other': 1.0*data['other']['hp']/self.data[trainer]['initial_info']['hp_other'],
                                             'isWin': data['isWin']}

        # 统计完成的pvp对局数，先获取互斥锁
        with self.finish_cnt_lock:
            self.finish_cnt += 1
        logger.info('trainer:{}, finish_info:{}'.format(
            trainer, str(self.data[trainer]['finish_info'])))

        #获得reward
        win_reward = 1 if self.data[trainer]['finish_info']['isWin'] else 0
        for i,state in enumerate(self.data[trainer]['states']):
            if i==(len(self.data[trainer]['states'])-1):
                self.data[trainer]['rewards'].append(0.5*win_reward+(0.1*(self.data[trainer]['finish_info']['hp_self']-state[0])+0.4*
                                                                 (state[1]-self.data[trainer]['finish_info']['hp_other'])))
            else:
                self.data[trainer]['rewards'].append(0.5*win_reward+(0.1*(self.data[trainer]['states'][i+1][0]-state[0])+0.4*
                                                                 (state[1]-self.data[trainer]['states'][i+1][1])))

        #构造(state,action,reward,next_state)对
        memory=self.get_memoryset(trainer)
        path={'observation':np.array(map(lambda x:x[0],memory)),'action':np.array(map(lambda x:x[1],memory)),
               'reward': np.array(map(lambda x:x[2],memory)), 'next_observation': np.array(map(lambda x:x[3],memory)),
              'prob':np.array(self.data[trainer]['prob'][:-1]),
               'isWin':data['isWin']}
        self.paths.append(path)

        if self.data[trainer]['finish_info']['isWin']:
            self.win_cnt+=1
            self.win_memory.extend(memory)
        else:
            self.lose_cnt+=1
            self.lose_memory.extend(memory)

        log_info = 'iter:{}, win_ratio:{}, trainer:{},  win_cnt:{}, lose_cnt:{} \n'.format(
            self.iter, 1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt), trainer, self.win_cnt, self.lose_cnt)

        logger.info(log_info)
        self.wpct_file.writelines(','.join([str(self.iter), str(1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt)), '\n']))

        # 如果开始的pvp都完成了
        if self.finish_cnt >= self.batch_size:
            self.wpct_file.flush()
            self.iter += 1

            self.compute_advantage(self.agent.baseline, self.paths, gamma=cfg["gamma"], lam=cfg["lam"])
            # VF Update ========
            self.agent.baseline.fit(self.paths)
            # Pol Update ========
            self.agent.updater(self.paths)

            if self.iter < self.iter_num:
                self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))

    def compute_advantage(self,vf, paths, gamma, lam):
        # Compute return, baseline, advantage
        for path in paths:
            path["return"] = discount(path["reward"], gamma)
            b = path["baseline"] = vf.predict(path)
            b1 = np.append(b, 0 if path["isWin"] else b[-1])
            deltas = path["reward"] + gamma * b1[1:] - b1[:-1]
            path["advantage"] = discount(deltas, gamma * lam)
            path["advantage"] =np.nan_to_num(path["advantage"])
        alladv = np.concatenate([path["advantage"] for path in paths])
        # Standardize advantage
        std = alladv.std()
        mean = alladv.mean()
        for path in paths:
            path["advantage"] = (path["advantage"] - mean) / std

