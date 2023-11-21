from .tag import PretrainedTag, RandomTag, FrozenTag, HeuristicTag, BullTag

REGISTRY = {
    "heuristic_tag": HeuristicTag,
    "pretrained_tag": PretrainedTag,
    "random_tag": RandomTag,
    "frozen_tag": FrozenTag,
    "bull_tag": BullTag,
}