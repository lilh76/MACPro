REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC

from .basic_controller_macpro_attn import BasicMACMACProAttn

REGISTRY["basic_mac_macpro_attn"] = BasicMACMACProAttn

from .basic_controller_macpro_rnn import BasicMACMACProRNN

REGISTRY["basic_mac_macpro_rnn"] = BasicMACMACProRNN