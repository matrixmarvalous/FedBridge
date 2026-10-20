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

from tianshou.trainer.fedonpolicy import (
    FedOnpolicyTrainer,
    fedonpolicy_trainer,
    fedonpolicy_trainer_iter,
)

from tianshou.trainer.fedavgonpolicy import (
    FedAvgOnpolicyTrainer,
    fedavgonpolicy_trainer,
    fedavgonpolicy_trainer_iter,
)


from tianshou.trainer.fedbridgeavg import (
    FedBridgeAvgTrainer,
    fedbridgeavg_trainer,
    fedbridgeavg_trainer_iter,
)

from tianshou.trainer.fedonpolicy_push_avg import (
    FedOnpolicy_PushAvgTrainer,
    fedonpolicy_push_avg_trainer,
    fedonpolicy_push_avg_trainer_iter,
)

from tianshou.trainer.perfeddc import (
    PerFedDCTrainer,
    perfeddc_trainer,
    perfeddc_trainer_iter,
)

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
