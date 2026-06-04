"""Model modules for HyperCP.

Modules
-------
- encoder       : DeepSets feature encoder + HGNN convolutions + TCN  (DONE)
- forecast_head : point-forecast head (Huber loss)                    (DONE)
- quantile_head : multi-quantile head (pinball loss)                  (DONE)
- conformity    : hyperedge-level non-conformity scores               (DONE)
- joint         : encoder + heads + FAMO/Kendall/equal weighting      (DONE)
"""

from .encoder import (
    DeepSetsFeatureEncoder,
    HypergraphConv,
    TemporalConvBlock,
    HyperCPEncoder,
    EncoderConfig,
)
from .forecast_head import (
    ForecastHead,
    ForecastHeadConfig,
    huber_forecast_loss,
)
from .quantile_head import (
    QuantileHead,
    QuantileHeadConfig,
    pinball_loss,
)
from .conformity import (
    HyperedgeConformityScore,
    ConformityConfig,
    pinball_residual,
    build_weight_matrix,
    hyperedge_cqr_score,
)
from .joint import (
    HyperCPJointModel,
    JointConfig,
    BaseWeighter,
    EqualWeighter,
    KendallWeighter,
    FAMOWeighter,
    make_weighter,
)

__all__ = [
    "DeepSetsFeatureEncoder",
    "HypergraphConv",
    "TemporalConvBlock",
    "HyperCPEncoder",
    "EncoderConfig",
    "ForecastHead",
    "ForecastHeadConfig",
    "huber_forecast_loss",
    "QuantileHead",
    "QuantileHeadConfig",
    "pinball_loss",
    "HyperedgeConformityScore",
    "ConformityConfig",
    "pinball_residual",
    "build_weight_matrix",
    "hyperedge_cqr_score",
    "HyperCPJointModel",
    "JointConfig",
    "BaseWeighter",
    "EqualWeighter",
    "KendallWeighter",
    "FAMOWeighter",
    "make_weighter",
]
