REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .macpro_rnn_agent import MACProRNNAgent
REGISTRY["macpro_rnn"] = MACProRNNAgent

from .attn_agent import AttnAgent
REGISTRY["attn"] = AttnAgent

from .macpro_attn_agent import MACProAttnAgent
REGISTRY["macpro_attn"] = MACProAttnAgent