from collections import OrderedDict
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from .trpo import TrpoUpdater
from .misc_utils import *
from .core import *
from .filters import *

MLP_OPTIONS = [
    ("hid_sizes", lambda x : map(int, x.split(",")) if x else [], [64,64], "Sizes of hidden layers of MLP"),
    ("activation", str, "tanh", "nonlinearity")
]

FILTER_OPTIONS = [
    ("filter", int, 0, "Whether to do a running average filter of the incoming observations and rewards")
]
class AgentWithPolicy(object):
    def __init__(self, policy, obfilter, rewfilter):
        self.policy = policy
        self.obfilter = obfilter
        self.rewfilter = rewfilter
        self.stochastic = True
    def set_stochastic(self, stochastic):
        self.stochastic = stochastic
    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic = self.stochastic)
    def get_flat(self):
        return self.policy.get_flat()
    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)
    def obfilt(self, ob):
        return self.obfilter(ob)
    def rewfilt(self, rew):
        return self.rewfilter(rew)

class TrpoAgent(AgentWithPolicy):
    # 6 is the ac_dim,9 is the ob_dim.
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS
    def __init__(self):
        cfg = update_default_config(self.options)
        policy, self.baseline = make_mlps(cfg)
        obfilter, rewfilter = make_filters(cfg)
        self.updater = TrpoUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

def make_mlps(cfg):
    hid_sizes = cfg["hid_sizes"]
    outdim = 6
    probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(9,)) if i==0 else {}
        net.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    net.add(Dense(outdim, activation="softmax"))
    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(9+1,)) if i==0 else {} # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, cfg["timestep_limit"], dict(mixfrac=0.1))
    return policy, baseline

def make_filters(cfg):
    if cfg["filter"]:
        obfilter = ZFilter(9, clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter

