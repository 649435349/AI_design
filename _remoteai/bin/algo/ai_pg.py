# -*- coding: utf-8 -*-

import numpy as np
import cPickle as pickle
import threading

from conf import configs
from utils import log4py
from collections import defaultdict
from operator import itemgetter
logger = log4py.Logger(name='AI train', log_file=configs.DEFAULT_LOGGER_FILE)

skill_ids = [3120, 3060, 3050, 3080, 3000]


def get_distance(pos1, pos2):
    """Calculate the distance of two positions"""
    pos1 = np.array(eval(pos1))
    pos2 = np.array(eval(pos2))
    return int(np.linalg.norm(pos1 - pos2))


def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


class pg_agent():
    """Policy Gradient"""

    def __init__(self, resume=False):

        self.resume = resume
        self.batch_size = 5
        self.iter_num = 1000
        self.iter = 0
        self.move_actions = [-50, 50]
        # reset some parameters
        self.reset_par()
        # initialize PG parameters
        self.pg_initialize()
        # running reward
        self.running_reward = None
        self.win_cnt = 0
        self.lose_cnt = 0
        self.wpct_file = open(configs.WPCT_FILE, 'wb')

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0
        self.finish_cnt_lock = threading.Lock()

    def pg_initialize(self):
        """初始化PG模型参数"""
        self.learning_rate = 1e-3
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.action_dim = 5
        self.state_dim = 3 + self.action_dim
        self.hidden_dim = 100

        if self.resume:
            self.model = pickle.load(open(configs.DEFAULT_PG_MODEL, 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(
                self.state_dim, self.hidden_dim) / np.sqrt(self.state_dim)  # "Xavier" initialization
            self.model['W2'] = np.random.randn(
                self.hidden_dim, self.action_dim) / np.sqrt(self.hidden_dim)
        # update buffers that add up gradients over a batch
        self.grad_buffer = {k: np.zeros_like(
            v, dtype=np.float) for k, v in self.model.iteritems()}
        # rmsprop memory
        self.rmsprop_cache = {k: np.zeros_like(
            v, dtype=np.float) for k, v in self.model.iteritems()}

    def startPVP(self, MQ_file=configs.DEFAULT_MQ_FILE):
        """将PVP双方信息写入MQ，开始一批PVP对局"""
        self.reset_par()  # 首先重置参数
        role_data = '{"op": "start", "AI": {"career": 6, "level": 35}, "BT": {"career": 6, "level": 35}}'
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
        self.data[trainer]['xs'] = []
        self.data[trainer]['hs'] = []
        self.data[trainer]['dlogps'] = []
        self.data[trainer]['drs'] = []

        self.data[trainer]['hp_self'] = 0
        self.data[trainer]['hp_other'] = 0


    def __format_action(self, action_ind, skill_ids, trainer):
        """将action编号转化为对应的格式化action数据"""
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

    def get_action(self, state):
        """根据环境状态信息输出action"""
        trainer = state['name']
        if trainer not in self.data:
            return

        # skill_ids = state['self']['skillids']
        # skill_ids = [1000, 1100, 1150, 1070, 1060, 1170]
        # skill_ids = [3120, 3060, 3050, 40001, 3000]
        # skill_ids = [2210, 2280, 2310, 2120, 2290, 2080]

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
                action_ind = np.random.choice(range(len(skill_ids) + 2))
                return self.__format_action(action_ind, skill_ids, trainer)

        self.data[trainer]['step_id'] += 1

        x = self.__format_state(state)
        

        aprob, h = self.policy_forward(x)

        # roll the dice, in the softmax loss
        u = np.random.uniform()
        aprob_cum = np.cumsum(aprob)
        action_ind = np.where(u <= aprob_cum)[0][0]

        # record various intermediates (needed later for backprop)
        self.data[trainer]['xs'].append(x)  # observation
        self.data[trainer]['hs'].append(h)  # hidden state
        # softmax loss gradient
        dlogsoftmax = aprob.copy()
        dlogsoftmax[0, action_ind] -= 1
        self.data[trainer]['dlogps'].append(dlogsoftmax)

        
        if self.data[trainer]['step_id'] > 1:
            reward = 1 if (self.data[trainer]['hp_other'] - x[1]) > (self.data[trainer]['hp_self'] - x[0]) else 0
            # reward = 0
            self.data[trainer]['drs'].append(reward)

        self.data[trainer]['hp_self'] = x[0]
        self.data[trainer]['hp_other'] = x[1]


        logger.info('trainer:{}, step_id:{}, action:{}'.format(
            trainer, self.data[trainer]['step_id'], action_ind))

        # logger.info('trainer:{}, step_id:{}, state_vec:{}, action:{}, act_prob:{}'.format(
        #     trainer, self.data[trainer]['step_id'], str(x), action_ind, str(aprob)))

        return self.__format_action(action_ind, skill_ids, trainer)

    def __format_state(self, state):
        """从request_action传入的原始data中提取需要的state数据"""

        print '\n'
        print state['self']['skillids']
        print state['self']['refuseSkillids']

        skill_available = state['self']['skillids']
        skill_state = [0 for i in range(self.action_dim)]
        for (i, skill) in enumerate(skill_ids):
            if skill in skill_available:
                skill_state[i] = 1
            if skill in state['self']['refuseSkillids'] and state['self']['refuseSkillids'][skill] == 'so far':
                skill_state[i] = 1
            if i == self.action_dim - 1:
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

    def discount_rewards_old(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r, dtype=np.float)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        if(len(x.shape) == 1):
            x = x[np.newaxis, ...]
        h = x.dot(self.model['W1'])
        h[h < 0] = 0  # ReLU nonlinearity
        logp = h.dot(self.model['W2'])
        p = softmax(logp)
        return p, h  # return probability of taking actions, and hidden state

    def policy_backward(self, epx, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = eph.T.dot(epdlogp)
        dh = epdlogp.dot(self.model['W2'].T)
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = epx.T.dot(dh)
        return {'W1': dW1, 'W2': dW2}

    def __update_model(self):
        # stack together all inputs, hidden states, action gradients, and
        # rewards for a batch of PVPs
        xs, hs, dlogps, drs_discount = [], [], [], []
        for trainer in self.data.iterkeys():
            xs += self.data[trainer]['xs']
            hs += self.data[trainer]['hs']
            dlogps += self.data[trainer]['dlogps']
            drs_discount += self.discount_rewards(self.data[trainer]['drs'])

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        discounted_epr = np.vstack(drs_discount)

        # standardize the rewards to be unit normal (helps control the gradient
        # estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = self.policy_backward(epx, eph, epdlogp)
        for k in self.model:
            self.grad_buffer[k] += grad[k]  # accumulate grad over batch

        for k, v in self.model.iteritems():
            g = self.grad_buffer[k]  # gradient
            self.rmsprop_cache[k] = self.decay_rate * \
                self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
            # 注意，softmax版这边是减号，原来的sigmoid版是+号 ！！！！！
            self.model[k] -= self.learning_rate * g / \
                (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            self.grad_buffer[k] = np.zeros_like(
                v, dtype=np.float)  # reset batch gradient buffer

    def add_finish_data(self, data):
        """获取onPVPFinish时传入的数据"""
        trainer = data['name']
        if trainer not in self.data:
            return

        self.data[trainer]['finish_info'] = {'hp_self': data['self']['hp'],
                                             'hp_other': data['other']['hp'],
                                             'isWin': data['isWin']}
        # 统计完成的pvp对局数
        with self.finish_cnt_lock:
            self.finish_cnt += 1
        logger.info('trainer:{}, finish_info:{}'.format(
            trainer, str(self.data[trainer]['finish_info'])))

        # 获取最终reward
        finish_reward = self.__get_reward(trainer)
        # self.data[trainer]['drs'][-1] = finish_reward
        self.data[trainer]['drs'].append(finish_reward)

        print len(self.data[trainer]['xs']), len(self.data[trainer]['hs']), len(self.data[trainer]['dlogps']), len(self.data[trainer]['drs'])

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
                self.__update_model()
                if (self.iter + 1) % 10 == 0:
                    pickle.dump(self.model, open(
                        configs.DEFAULT_PG_MODEL, 'wb'))
                # self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))
