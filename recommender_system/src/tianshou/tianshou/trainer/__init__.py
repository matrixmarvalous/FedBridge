"""Trainer package."""

from tianshou.trainer.base import BaseTrainer
from tianshou.trainer.fedbase import FedBaseTrainer
from tianshou.trainer.offline import (
    OfflineTrainer,
    offline_trainer,
    offline_trainer_iter,
)
from tianshou.trainer.offpolicy import (
    OffpolicyTrainer,
    offpolicy_trainer,
    offpolicy_trainer_iter,
)
from tianshou.trainer.onpolicy import (
    OnpolicyTrainer,
    onpolicy_trainer,
    onpolicy_trainer_iter,
)

from tianshou.trainer.fedonpolicy import (  # 文件名需与实际一致
    FedOnpolicyTrainer,
    fedonpolicy_trainer,
    fedonpolicy_trainer_iter,
)

from tianshou.trainer.fedavgonpolicy import (  # 文件名需与实际一致
    FedAvgOnpolicyTrainer,
    fedavgonpolicy_trainer,
    fedavgonpolicy_trainer_iter,
)


# Codex-modified 2026-08-18: the canonical Bridge PPO route is
# fedbridgeonp below; remove the duplicated pre-release trainer export.
from tianshou.trainer.fedbridgeavg import (  # 文件名需与实际一致
    FedBridgeAvgTrainer,
    fedbridgeavg_trainer,
    fedbridgeavg_trainer_iter,
)

from tianshou.trainer.fedonpolicy_pushpull import (  # 文件名需与实际一致
    FedOnpolicy_PushPullTrainer,
    fedonpolicy_pushpulltrainer,
    fedonpolicy_pushpulltrainer_iter,
)

from tianshou.trainer.perfeddc import (
    PerFedDCTrainer,
    perfeddc_trainer,
    perfeddc_trainer_iter,
)

# Codex-added 2026-05-19: pFedMe trainer export for EasyRL4Rec recommendation experiments.
from tianshou.trainer.pfedme import (
    pFedMeTrainer,
    pfedme_trainer,
    pfedme_trainer_iter,
)

from tianshou.trainer.fedbridgeoffp import (
    FedBridgeOffPTrainer,
    fedbridgeoffp_trainer,
    fedbridgeoffp_trainer_iter,
)

from tianshou.trainer.fedbridgeonp import (
    FedBridgeOnPTrainer,
    fedbridgeonp_trainer,
    fedbridgeonp_trainer_iter,
)

from tianshou.trainer.fedbridgehoffp import (
    FedBridgeHOffPTrainer,
    fedbridgehoffp_trainer,
    fedbridgehoffp_trainer_iter,
)

from tianshou.trainer.fedbridgehonp import (
    FedBridgeHOnPTrainer,
    fedbridgehonp_trainer,
    fedbridgehonp_trainer_iter,
)


from tianshou.trainer.fedoffpolicy import (
    FedOffpolicyTrainer,
    fedoffpolicy_trainer,
    fedoffpolicy_trainer_iter,
)



from tianshou.trainer.utils import gather_info, test_episode

__all__ = [
    "BaseTrainer",
    'FedBaseTrainer',
    "offpolicy_trainer",
    "offpolicy_trainer_iter",
    "OffpolicyTrainer",
    "onpolicy_trainer",
    "onpolicy_trainer_iter",
    "OnpolicyTrainer",
    "offline_trainer",
    "offline_trainer_iter",
    "OfflineTrainer",
    'FedOnpolicyTrainer',
    'fedonpolicy_trainer',
    'fedonpolicy_trainer_iter',
    'PerFedDCTrainer',
    'perfeddc_trainer',
    'perfeddc_trainer_iter',
    # Codex-added 2026-05-19: pFedMe baseline aligned with MuJoCo comparisons.
    'pFedMeTrainer',
    'pfedme_trainer',
    'pfedme_trainer_iter',
    'FedBridgeOffPTrainer',
    'fedbridgeoffp_trainer',
    'fedbridgeoffp_trainer_iter',
    'FedBridgeOnPTrainer',
    'fedbridgeonp_trainer',
    'fedbridgeonp_trainer_iter',
    'FedBridgeHOffPTrainer',
    'fedbridgehoffp_trainer',
    'fedbridgehoffp_trainer_iter',
    'FedBridgeHOnPTrainer',
    'fedbridgehonp_trainer',
    'fedbridgehonp_trainer_iter',
    'FedOffpolicyTrainer',
    'fedoffpolicy_trainer',
    'fedoffpolicy_trainer_iter',
    "test_episode",
    "gather_info",
]
