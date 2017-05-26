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
skill_ids={'1':[1100, 1150, 1070, 1060, 1170, 1000],'6':[3080,3060 ,3070, 3140, 3100, 3000],'5':[2210, 2280, 2310, 2120, 2290, 2080]}
hps={'1':2007,'6':987,'5':987}

def get_distance(pos1, pos2):
    """计算两点之间距离"""
    pos1 = np.array(eval(pos1))
    pos2 = np.array(eval(pos2))
    return int(np.linalg.norm(pos1 - pos2))

class mc_agent_dqn_fyf():
    '''
    DQN
    '''
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

        # 初始化模型参数
        self.mc_initialize()

        # reward
        self.win_cnt = 0
        self.lose_cnt = 0

        #赢得太少了,记录所有赢的memory对
        self.win_memory=deque([],300)

        self.wpct_file = open(configs.WPCT_FILE, 'wb+')
        #self.model_file = open('../../model/'+self.who_vs_who, 'wb+')

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0
        # 为finish_cnt上锁，防止同时访问
        self.finish_cnt_lock = threading.Lock()

    def mc_initialize(self):
        """初始化模型参数"""
        self.learning_rate = 1e-1
        self.gamma = 0.9      # discount factor for reward

        #随机搜索的概率
        self.epsi_start = 0 if self.resume else 0.5
        self.epsi_end = 0

        self.action_dim = len(skill_ids[self.who_vs_who[0]])
        self.state_dim =2+1+len(skill_ids[self.who_vs_who[0]])

        if self.resume:
            self.model=load_model('../../model/dqn'+self.who_vs_who+'.h5')
        else:
            self.model = Sequential()
            self.model.add(Dense(128, activation='sigmoid', input_dim=self.state_dim,kernel_initializer=glorot_normal(2017)))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='sigmoid',kernel_initializer=glorot_normal(2017)))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.action_dim, activation='linear'))
            self.model.compile(loss='mse',
                          optimizer=SGD(lr=self.learning_rate))

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
        self.data[trainer]['initial_info'] = {'hp_self': data['self']['hp'],
                                              'hp_other': data['other']['hp'] if data['other'] else hps[self.who_vs_who[1]],
                                              'dist': get_distance(data['self']['position'],
                                                                   data['other']['position']) if data[
                                                  'other'] else 500
                                              }
        can_action, now_state = self.__format_state(data)
        self.data[trainer]['states'].append(now_state)

        self.data[trainer]['step_id'] += 1

        action_ind = self.get_epsi_greedy_action(can_action, now_state)

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
                skill_state[i] = 2
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
        state_vec = np.array([hp_self, hp_other] +[dist] + skill_state)
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

        action_ind = self.get_epsi_greedy_action(can_action,now_state)

        self.data[trainer]['actions'].append(action_ind)
        logger.info('trainer:{}, step_id:{}, action_id:{}'.format(
            trainer, self.data[trainer]['step_id'], action_ind))

        return self.__format_action(action_ind, trainer)

    def get_epsi_greedy_action(self,can_action,now_state):
        # 随机采样的概率
        epsi =self.epsi_start-1.0*self.iter / self.iter_num * (self.epsi_start - self.epsi_end)
        if random.random() < epsi:
            return np.random.choice(range(self.action_dim))
        else:
            val = self.model.predict(np.array([now_state]))
            act = np.argsort(val[0])
            for i in act[::-1]:
                if can_action[i]:
                    return i

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
        res.append((self.data[trainer]['states'][len1-1], self.data[trainer]['actions'][len1-1],
                    self.data[trainer]['rewards'][len1-1],(self.data[trainer]['finish_info']['hp_self'],self.data[trainer]['finish_info']['hp_other'])))
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
        if self.data[trainer]['finish_info']['isWin']:
            self.win_cnt+=1
        else:
            self.lose_cnt+=1
        # 统计完成的pvp对局数，先获取互斥锁
        with self.finish_cnt_lock:
            self.finish_cnt += 1
        logger.info('trainer:{}, finish_info:{}'.format(
            trainer, str(self.data[trainer]['finish_info'])))

        #获得reward
        win_reward = 10 if self.data[trainer]['finish_info']['isWin'] else 1
        for i,state in enumerate(self.data[trainer]['states']):
            if i==(len(self.data[trainer]['states'])-1):
                self.data[trainer]['rewards'].append(win_reward*(#0.1*(self.data[trainer]['finish_info']['hp_self']-state[0])+0.9*
                                                                 (state[1]-self.data[trainer]['finish_info']['hp_other'])))
            else:
                tmp=0
                for j,state2 in enumerate(self.data[trainer]['states'][(i+1):]):
                    if j!=(len(self.data[trainer]['states'])-i-1-1):
                        tmp+=(self.gamma**j)*(win_reward*(#0.1*(state2[0]-state[0])+0.9*
                                                                     (state[1]-state2[1])))
                    else:
                        tmp+=(self.gamma**j)*(win_reward*(#0.1*(state2[0]-state[0])+0.9*
                                                                     (state[1]-state2[1])))
                self.data[trainer]['rewards'].append(tmp)


        #构造(state,action,reward,next_state)对
        memory=self.get_memoryset(trainer)
        if self.data[trainer]['finish_info']['isWin']:
            self.win_memory.extend(memory)

        log_info = 'iter:{}, win_ratio:{}, trainer:{},  win_cnt:{}, lose_cnt:{} \n'.format(
            self.iter, 1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt), trainer, self.win_cnt, self.lose_cnt)

        logger.info(log_info)
        self.wpct_file.writelines(','.join([str(self.iter), str(1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt)), '\n']))

        #更新模型
        #普攻实在太多了，搞少一点吧
        try:
            random_set_no5=random.sample(filter(lambda x :x[1]!=5,memory),20)
            random_set_5=random.sample(filter(lambda x :x[1]==5,memory),4)
        except:
            random_set_no5 = random.sample(filter(lambda x: x[1] != 5, memory), 5)
            random_set_5 = random.sample(filter(lambda x: x[1] == 5, memory), 1)
        random_set=random_set_no5+random_set_5
        #random_set=random.sample(memory,int(0.3*len(memory)))
        features, labels = [], []

        for state,action,reward,next_state in random_set:
            try:
                tmp = self.model.predict(np.array([state]))
                tmp[0][int(action)] = self.gamma * tmp[0][int(action)] + (1.0 - self.gamma) * (
                    reward + max(self.model.predict(np.array([next_state]))[0]))
                features.append(copy.deepcopy(state))
                labels.append(copy.deepcopy(tmp[0]))
            except:
                continue

        self.model.fit(np.array(features),np.array(labels),batch_size=10,epochs=1)

        # 如果开始的pvp都完成了
        if self.finish_cnt >= self.batch_size:
            self.wpct_file.flush()
            self.iter += 1
            '''
            # 强化胜利
            if self.win_memory:
                random_set_no5 = random.sample(filter(lambda x: x[1] != 5, self.win_memory), int(0.1*len(filter(lambda x: x[1] != 5, self.win_memory)))
                random_set_5 = random.sample(filter(lambda x: x[1] == 5, self.win_memory), 4)
                win_set = random_set_no5+random_set_5

                features, labels = [], []

                for state, action, reward, next_state in win_set:
                    try:
                        tmp = self.model.predict(np.array([state]))
                        tmp[0][int(action)] = self.gamma * tmp[0][int(action)] + (1.0 - self.gamma) * (
                            reward + max(self.model.predict(np.array([next_state]))[0]))
                        features.append(copy.deepcopy(state))
                        labels.append(copy.deepcopy(tmp[0]))
                    except:
                        continue
                self.model.fit(np.array(features), np.array(labels), batch_size=10, epochs=1)
            '''
            if self.iter < self.iter_num:
                # 每次迭代，也就是5次PVP，更新模型
                self.model.save('../../model/'+self.who_vs_who+'.h5')
                self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))