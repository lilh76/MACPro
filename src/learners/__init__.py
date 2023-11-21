from .q_learner import QLearner
from .q_learner_macpro_attn import QLearnerMACProAttn
from .q_learner_macpro_rnn import QLearnerMACProRNN

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_macpro_attn"] = QLearnerMACProAttn
REGISTRY["q_learner_macpro_rnn"] = QLearnerMACProRNN
