# -*- coding: utf-8 -*-

import numpy as np
import cPickle as pickle

from conf import configs
from utils import log4py
from collections import defaultdict
from operator import itemgetter
logger = log4py.Logger(name='AI train', log_file=configs.DEFAULT_LOGGER_FILE)


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
        self.iter_num = 100
        self.iter = 0
        self.move_actions = [-50, 50]
        # reset some parameters
        self.reset_par()
        # initialize PG parameters
        self.pg_initialize()
        # running reward
        self.running_reward = None

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0

    def pg_initialize(self):
        """初始化PG模型参数"""
        self.learning_rate = 1e-3
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.state_dim = 3
        self.hidden_dim = 100
        self.action_dim = 5
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
        role_data = '{"op": "start", "AI": {"career": 1, "level": 30}, "BT": {"career": 1, "level": 30}}'
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
        if 'initial_info' not in self.data[trainer]:
            # 获取初始化信息
            self.data[trainer]['initial_info'] = {'hp_self': state['self']['hp'],
                                                  'hp_other': state['other']['hp'],
                                                  'dist': get_distance(state['self']['position'], state['other']['position'])}

        skill_ids = state['self']['skillids']
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
        dlogsoftmax[0, action_ind] -= 1  # -discounted reward
        self.data[trainer]['dlogps'].append(dlogsoftmax)

        # 暂时先不考虑中间步骤的reward
        reward = 0.0
        self.data[trainer]['drs'].append(reward)

        logger.info('trainer:{}, step_id:{}, state_vec:{}, action:{}, act_prob:{}'.format(
            trainer, self.data[trainer]['step_id'], str(x), action_ind, str(aprob)))
        return self.__format_action(action_ind, skill_ids, trainer)

    def __format_state(self, state):
        """从request_action传入的原始data中提取需要的state数据"""
        trainer = state['name']
        hp_self = 1.0 * state['self']['hp'] / \
            self.data[trainer]['initial_info']['hp_self']
        hp_other = 1.0 * state['other']['hp'] / \
            self.data[trainer]['initial_info']['hp_other']
        dist = 1.0 * get_distance(state['self']['position'], state['other'][
                                  'position']) / self.data[trainer]['initial_info']['dist']
        state_vec = np.array([hp_self, hp_other, dist])
        return state_vec

    def __get_reward(self, trainer):
        initial_info = self.data[trainer]['initial_info']
        finish_info = self.data[trainer]['finish_info']
        reward = 1.0 * finish_info['hp_self'] / initial_info['hp_self'] - \
            1.0 * finish_info['hp_other'] / initial_info['hp_other']
        if finish_info['isWin']:
            reward += 0.0  # 修改为不增加胜利后的reward
        if reward == 0:  # 临时应对reward为0时发生的除0问题
            reward = 0.1
        return reward

    def discount_rewards(self, r):
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
        self.data[trainer]['finish_info'] = {'hp_self': data['self']['hp'],
                                             'hp_other': data['other']['hp'],
                                             'isWin': data['isWin']}
        # 统计完成的pvp对局数
        self.finish_cnt += 1
        logger.info('trainer:{}, finish_info:{}'.format(
            trainer, str(self.data[trainer]['finish_info'])))

        # 获取最终reward
        finish_reward = self.__get_reward(trainer)
        self.data[trainer]['drs'][-1] = finish_reward

        # stack together all inputs, hidden states, action gradients, and
        # rewards for this pvp
        epx = np.vstack(self.data[trainer]['xs'])
        eph = np.vstack(self.data[trainer]['hs'])
        epdlogp = np.vstack(self.data[trainer]['dlogps'])
        epr = np.vstack(self.data[trainer]['drs'])

        # compute the discounted reward backwards through time
        discounted_epr = self.discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        # 单局计算grad的时候可以考虑不normalize奖励，待尝试，因为我们希望model往正奖励方向（和负奖励反方向）改进，如果normalize的话不管胜负，一局内总是有奖励为正，有奖励为负
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = self.policy_backward(epx, eph, epdlogp)
        for k in self.model:
            self.grad_buffer[k] += grad[k]  # accumulate grad over batch

        # boring book-keeping
        self.running_reward = finish_reward if self.running_reward is None else self.running_reward * \
            0.99 + finish_reward * 0.01
        logger.info('iter:{}, trainer:{}, reward:{}, running_reward:{}'.format(
            self.iter, trainer, finish_reward, self.running_reward))

        # 如果开始的pvp都完成了，perform rmsprop parameter update, 更新模型参数
        if self.finish_cnt >= self.batch_size:
            self.iter += 1
            if self.iter < self.iter_num:
                self.__update_model()
                if self.iter % 10 == 0:
                    pickle.dump(self.model, open(
                        configs.DEFAULT_PG_MODEL, 'wb'))
                self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))


class cem_agent():
    """ Noisy Cross Entropy Method"""

    def __init__(self, batch_size=20, iter_num=20):

        self.batch_size = batch_size
        self.iter_num = iter_num
        self.iter = 0
        self.state_dim = 3
        self.action_dim = 5
        self.move_actions = [-50, 50]
        # reset some parameters
        self.reset_par()
        # initialize CEM parameters
        self.cem_initialize()

    def reset_par(self):
        self.data = defaultdict(defaultdict)
        self.start_cnt = 0
        self.finish_cnt = 0

    def cem_initialize(self):
        """初始化CEM算法参数"""
        self.par_mean = np.random.rand(
            self.state_dim * self.action_dim) * 2 - 1
        self.par_var = np.ones(self.par_mean.shape)
        self.top_ratio = 0.25

    def startPVP(self, MQ_file=configs.DEFAULT_MQ_FILE):
        """将PVP双方信息写入MQ，开始一批PVP对局"""
        role_data = '{"op": "start", "AI": {"career": 1, "level": 30}, "BT": {"career": 1, "level": 30}}'
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
        logger.info('iter:{}, start_cnt:{}, trainer:{}'.format(
            self.iter, self.start_cnt, trainer))

    def add_finish_data(self, data):
        """获取onPVPFinish时传入的数据"""
        trainer = data['name']
        self.data[trainer]['finish_info'] = {'hp_self': data['self']['hp'],
                                             'hp_other': data['other']['hp'],
                                             'isWin': data['isWin']}
        # 统计完成的pvp对局数
        self.finish_cnt += 1
        logger.info('iter:{}, finish_cnt:{}, trainer:{}, finish_info:{}'.format(
            self.iter, self.finish_cnt, trainer, str(self.data[trainer]['finish_info'])))
        # 如果开始的pvp都完成了，更新模型参数
        if self.finish_cnt >= self.batch_size:
            self.iter += 1
            if self.iter < self.iter_num:
                self.__update_model()
                self.reset_par()
                self.startPVP()
            else:
                logger.info('{} iterations finished!'.format(self.iter))

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
        if 'initial_info' not in self.data[trainer]:
            # 获取初始化信息
            self.data[trainer]['initial_info'] = {'hp_self': state['self']['hp'],
                                                  'hp_other': state['other']['hp'],
                                                  'dist': get_distance(state['self']['position'], state['other']['position'])}

        skill_ids = state['self']['skillids']
        self.data[trainer]['step_id'] += 1
        state = self.__format_state(state)
        if 'cem_par' in self.data[trainer]:
            cem_par = self.data[trainer]['cem_par'].reshape(
                self.action_dim, self.state_dim)
        else:
            # 初始化生成trainer对应的模型参数
            self.data[trainer]['cem_par'] = np.random.multivariate_normal(
                self.par_mean, np.diag(self.par_var), 1)
            cem_par = self.data[trainer]['cem_par'].reshape(
                self.action_dim, self.state_dim)

        action_ind = np.argmax(np.dot(cem_par, state))
        logger.info('trainer:{}, step_id:{}, state_vec:{}, action:{}'.format(
            trainer, self.data[trainer]['step_id'], str(state), action_ind))
        return self.__format_action(action_ind, skill_ids, trainer)

    def create_model(self):
        pass

    def __format_state(self, state):
        """从request_action传入的原始data中提取需要的state数据"""
        hp_self = state['self']['hp']
        hp_other = state['other']['hp']
        dist = get_distance(state['self']['position'],
                            state['other']['position'])
        state_vec = np.array([hp_self, hp_other, dist])
        return state_vec

    def __get_reward(self, trainer):
        initial_info = self.data[trainer]['initial_info']
        finish_info = self.data[trainer]['finish_info']
        reward = 1.0 * finish_info['hp_self'] / initial_info['hp_self'] - \
            1.0 * finish_info['hp_other'] / initial_info['hp_other']
        if finish_info['isWin']:
            reward += 1.0
        return reward

    def __update_model(self):
        trainer_rewards = dict()
        for trainer in self.data.iterkeys():
            trainer_rewards[trainer] = self.__get_reward(trainer)
        trainer_rewards_sorted = sorted(
            trainer_rewards.iteritems(), key=itemgetter(1), reverse=True)
        top_num = max(int(self.top_ratio * self.batch_size), 1)
        top_trainers = zip(*trainer_rewards_sorted[:top_num])[0]
        top_pars = [self.data[trainer]['cem_par'] for trainer in top_trainers]
        self.par_mean = np.mean(top_pars, axis=0)[0]
        self.par_var = np.var(top_pars, axis=0)[
            0] + np.ones(self.state_dim * self.action_dim) * max(0.1 - 0.01 * self.iter, 0)  # nosiy version
        logger.info('------------ iter_batch:{}, Mean top rewards:{} -------------'.format(
            self.iter, np.mean(zip(*trainer_rewards_sorted[:top_num])[1])))
