REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_runner_macpro import EpisodeRunnerMACPro
REGISTRY["episode_macpro"] = EpisodeRunnerMACPro


