# -*- coding: utf-8 -*-

import numpy as np
import pickle
import random
import threading
from sklearn.neural_network import MLPRegressor

from conf import configs
from utils import log4py
from collections import defaultdict
from operator import itemgetter
logger = log4py.Logger(name='AI train', log_file=configs.DEFAULT_LOGGER_FILE)


# BT使用的技能列表
# skill_ids = [3120, 3060, 3050, 3080, 3000]#猎人
# skill_ids = [2210, 2280, 2310, 2120, 2290, 2080]#牧师
skill_ids = [1100, 1150, 1070, 1060, 1170, 1000]#战士

def get_distance(pos1, pos2):
    """Calculate the distance of two positions"""
    pos1 = np.array(eval(pos1))
    pos2 = np.array(eval(pos2))
    return int(np.linalg.norm(pos1 - pos2))


class mc_agent():
    """Monte Carlo"""

    def __init__(self, resume=False):

        self.resume = resume
        self.batch_size = 5
        self.iter_num = 2000
        self.iter = 0
        self.move_actions = [-30, 30]
        # reset some parameters
        self.reset_par()
        # initialize MC parameters
        self.mc_initialize()
        # running reward
        self.running_reward = None
        self.win_cnt = 0
        self.lose_cnt = 0
        self.wpct_file = open(configs.WPCT_FILE, 'wb')
        self.model_file = open(configs.DEFAULT_MC_MODEL, 'wb')

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0
        # 为finish_cnt上锁，防止同时访问
        self.finish_cnt_lock = threading.Lock()

    def mc_initialize(self):
        """初始化MC模型参数"""
        self.learning_rate = 1e-3
        self.gamma = 0.9  # discount factor for reward
        self.epsi = 0.2
        self.epsi_start = 0.25
        self.epsi_end = 0.05
        self.action_dim = 6
        self.state_dim = 3 + 6

        if self.resume:
            self.model = pickle.load(open(configs.DEFAULT_MC_MODEL, 'rb'))
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(100))
            self.model.partial_fit(np.random.rand(
                self.state_dim + self.action_dim, 1).T, [0])

    def startPVP(self, MQ_file=configs.DEFAULT_MQ_FILE):
        """将PVP双方信息写入MQ，开始一批PVP对局"""
        self.reset_par()  # 首先重置参数

        # 1000局之后对局为战士vs猎人
        if self.iter >= 1000:
            role_data = '{"op": "start", "AI": {"career": 1, "level": 35}, "BT": {"career": 5, "level": 35}}'
        else:
            role_data = '{"op": "start", "AI": {"career": 1, "level": 35}, "BT": {"career": 1, "level": 35}}'


        with open(MQ_file, 'w') as f:
            for i in range(self.batch_size):
                f.write(role_data + '\n')

    def add_start_data(self, data):
        """获取onPVPStart时传入的数据,
        因为有时会收不到other数据，所以initial_info移到request_action处获取"""
        trainer = data['name']
        self.data[trainer]['step_id'] = 0
        # 统计开始的pvp对局数
        self.start_cnt += 1
        # 用于存储对局中间数据
        self.data[trainer]['states'] = []
        self.data[trainer]['actions'] = []
        self.data[trainer]['rewards'] = []

        self.data[trainer]['hp_self'] = 0
        self.data[trainer]['hp_other'] = 0

        self.data[trainer]['cnt'] = 0

    def __format_action(self, action_ind, skill_ids, trainer):
        """
        将action编号转化为对应的格式化action数据
        action_id:1为放技能,2为移动.
        """

        if action_ind < len(skill_ids):
            action_data = {"trainer_name": trainer, "step_id": self.data[trainer][
                'step_id'], "action_parameters": {"skill_id": skill_ids[action_ind]}, "action_id": 1}
        elif action_ind < len(skill_ids) + len(self.move_actions):
            action_ind = action_ind - len(skill_ids)
            action_data = {"trainer_name": trainer, "step_id": self.data[trainer][
                'step_id'], "action_parameters": {"move_distance": self.move_actions[action_ind]}, "action_id": 2}
        else:
            raise ValueError(
                'action_ind:{} is out of range!'.format(action_ind))
        return action_data

    def get_feature_vector(self, f):
        """将状态和动作组合为特征向量
        """
        state = f[0]
        idx = f[1]
        f_v = [0.0 for i in range(self.action_dim)]
        f_v[idx] = 1
        for i in state:
            f_v.append(i)
        return f_v

    def get_opt_action_list(self, model, obs):
        '''获取每个技能的预期奖赏
        '''
        f = []
        for i in range(self.action_dim):
            f.append(self.get_feature_vector((obs, i)))
        val = model.predict(f)
        return val

    def get_opt_action(self, model, obs):
        '''选取值最大的技能'''
        f = []
        for i in range(self.action_dim):
            f.append(self.get_feature_vector((obs, i)))
        val = model.predict(f)
        act = np.argmax(val)  # choose the action with the larger Q-value
        return act

    def get_epsi_greedy_action(self, model, obs):
        # 随机采样的概率
        epsi = self.epsi_start - self.iter / 1000 * (self.epsi_start - self.epsi_end)
        if self.iter >= 1000:
            epsi = self.epsi_start - (self.iter - 1000) / 1000 * (self.epsi_start - self.epsi_end)

        if random.random() < epsi:
            return np.random.choice(range(self.action_dim))
        else:
            return self.get_opt_action(model, obs)

    def get_action(self, state):
        """根据环境状态信息输出action"""
        trainer = state['name']
        if trainer not in self.data:
            return

        if 'initial_info' not in self.data[trainer]:
            if state['other']:
                # 获取初始化信息
                self.data[trainer]['initial_info'] = {'hp_self': state['self']['hp'],
                                                      'hp_other': state['other']['hp'],
                                                      # 'dist':get_distance(state['self']['position'], state['other']['position'])
                                                      'dist': 500
                                                      }
            else:
                # 如果other字段的信息还没有，直接返回随机动作，step_id不增加
                action_ind = np.random.choice(range(self.action_dim))
                return self.__format_action(action_ind, skill_ids, trainer)

        self.data[trainer]['step_id'] += 1
        x = self.__format_state(state)


        action_ind = self.get_epsi_greedy_action(self.model, x)


        self.data[trainer]['states'].append(x)  # observation
        self.data[trainer]['actions'].append(action_ind)

        if self.data[trainer]['step_id'] > 1:
            # 奖赏估价函数
            reward = 1 if (self.data[trainer]['hp_other'] - x[1]) > (self.data[trainer]['hp_self'] - x[0]) else 0
            self.data[trainer]['rewards'].append(reward)
        self.data[trainer]['hp_self'] = x[0]
        self.data[trainer]['hp_other'] = x[1]


        # # 获取可用技能列表，so far设置为可用，普攻一直可用
        # skill_available = state['self']['skillids']
        # skill_state = [0 for i in range(self.action_dim)]
        # for (i, skill) in enumerate(skill_ids):
        #     if skill in skill_available:
        #         skill_state[i] = 1
        #     if skill in state['self']['refuseSkillids'] and state['self']['refuseSkillids'][skill] == 'so far':
        #         skill_state[i] = 1
        #     if i == self.action_dim - 1:
        #         skill_state[i] = 1

        # # 按概率选取，并跳过不能释放的技能
        # action_prob = self.get_opt_action_list(self.model, x)
        # sort_index = np.argsort(action_prob)
        # for index in np.flipud(sort_index):
        #     if skill_state[index] == 1:
        #         action_ind = index
        #         break

        # 循环选取可用技能
        # action_ind = None
        # for i in range(5):
        #     self.data[trainer]['cnt'] = (self.data[trainer]['cnt'] + 1) % 5
        #     if skill_state[self.data[trainer]['cnt']] == 1:
        #         action_ind = self.data[trainer]['cnt']
        #         break
        # if action_ind == None:
        #     action_ind = 5


        logger.info('trainer:{}, step_id:{}, action_id:{}'.format(
            trainer, self.data[trainer]['step_id'], action_ind))

        return self.__format_action(action_ind, skill_ids, trainer)

    def __format_state(self, state):
        """从request_action传入的原始data中提取需要的state数据"""

        #print(state)

        #print '\n'
        #print state['self']['skillids']
        #print state['self']['refuseSkillids']

        # 获取可用技能列表，so far设置为可用，普攻一直可用
        skill_available = state['self']['skillids']
        skill_state = [0 for i in range(len(skill_ids))]
        for (i, skill) in enumerate(skill_ids):
            if skill in skill_available:
                skill_state[i] = 1
            if skill in state['self']['refuseSkillids'] and (state['self']['refuseSkillids'][skill] == 'so far'):
                skill_state[i] = 1
            if i == len(skill_ids) - 1:
                skill_state[i] = 1

        trainer = state['name']
        hp_self = 1.0 * state['self']['hp'] / \
            self.data[trainer]['initial_info']['hp_self']
        hp_other = 1.0 * state['other']['hp'] / \
            self.data[trainer]['initial_info']['hp_other']
        dist = 1.0 * get_distance(state['self']['position'], state['other'][
                                  'position']) / self.data[trainer]['initial_info']['dist']
        state_vec = np.array([hp_self, hp_other, dist] + skill_state)
        return state_vec

    def __get_reward(self, trainer):
        initial_info = self.data[trainer]['initial_info']
        finish_info = self.data[trainer]['finish_info']
        reward = 1.0 * finish_info['hp_self'] / initial_info['hp_self'] - \
            1.0 * finish_info['hp_other'] / initial_info['hp_other']

        # reward 计算方法
        reward = reward * 10

        if finish_info['isWin']:
            reward += 0.0  # 修改为不增加胜利后的reward
            self.win_cnt += 1
        else:
            self.lose_cnt += 1
        if reward == 0:  # 临时应对reward为0时发生的除0问题
            reward = 0.01
        return reward

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros(len(r))
        running_add = 0
        for t in reversed(xrange(0, len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r.tolist()

    def __update_model(self):
        # stack together all states, actions, and rewards for a batch of PVPs
        states, actions, discount_rewards = [], [], []
        for trainer in self.data.iterkeys():
            states += self.data[trainer]['states']
            actions += self.data[trainer]['actions']
            discount_rewards += self.discount_rewards(
                self.data[trainer]['rewards'])
        features = list(map(self.get_feature_vector, [
                        (states[i], actions[i]) for i in range(len(states))]))
        self.model.partial_fit(features, discount_rewards)

    def add_finish_data(self, data):
        """获取onPVPFinish时传入的数据"""
        trainer = data['name']
        if trainer not in self.data:
            return

        self.data[trainer]['finish_info'] = {'hp_self': data['self']['hp'],
                                             'hp_other': data['other']['hp'],
                                             'isWin': data['isWin']}

        # 统计完成的pvp对局数，先获取互斥锁
        with self.finish_cnt_lock:
            self.finish_cnt += 1
        logger.info('trainer:{}, finish_info:{}'.format(
            trainer, str(self.data[trainer]['finish_info'])))

        # 获取最终reward
        finish_reward = self.__get_reward(trainer)
        # self.data[trainer]['rewards'][-1] = finish_reward
        self.data[trainer]['rewards'].append(finish_reward)

        print len(self.data[trainer]['states']), len(self.data[trainer]['actions']), len(self.data[trainer]['rewards'])

        # boring book-keeping
        self.running_reward = finish_reward if self.running_reward is None else self.running_reward * \
            0.95 + finish_reward * 0.05

        log_info = 'iter:{}, win_ratio:{}, trainer:{}, reward:{}, running_reward:{}, win_cnt:{}, lose_cnt:{} \n'.format(
            self.iter, 1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt), trainer, finish_reward, self.running_reward, self.win_cnt, self.lose_cnt)

        logger.info(log_info)
        self.wpct_file.writelines(','.join([str(self.iter), str(
            finish_reward), str(self.running_reward), str(1.0 * self.win_cnt / (self.win_cnt + self.lose_cnt)), '\n']))

        # 如果开始的pvp都完成了，perform rmsprop parameter update, 更新模型参数
        if self.finish_cnt >= self.batch_size:
            self.wpct_file.flush()
            self.iter += 1
            if self.iter < self.iter_num:
                # self.__update_model()
                if (self.iter + 1) % 10 == 0:
                    pickle.dump(self.model, self.model_file)

                # 1000局之后改为战士vs猎人
                if self.iter == 1000:
                    self.running_reward = None
                    self.win_cnt = 0
                    self.lose_cnt = 0
                    self.model = MLPRegressor(hidden_layer_sizes=(100))
                    self.model.partial_fit(np.random.rand(
                        self.state_dim + self.action_dim, 1).T, [0])
                    self.model_file = open('../../resources/mc1.model', 'wb')

                self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))

            
