"""Conformal calibration modules.

Modules
-------
- aci           : Adaptive Conformal Inference (Gibbs-Candès 2021)      (DONE)
- split_cp      : standard split CP / CQR baseline                      (DONE)
- conformal_pid : Conformal PID Control (Angelopoulos NeurIPS 2023)     (DONE)
- agaci         : Aggregated ACI with BOA mixing (Zaffran ICML 2022)    (DONE)
- enbpi         : Ensemble-bootstrap PI (Xu & Xie ICML 2021)            (FUTURE -- bootstrap is invasive)
- encqr         : Ensemble CQR (Jensen 2022)                            (FUTURE)
"""

from .aci import ACI, ACIConfig
from .split_cp import SplitCP, SplitCPConfig
from .conformal_pid import ConformalPID, ConformalPIDConfig
from .agaci import AgACI, AgACIConfig
from .cf_gnn import CFGNNSmoothedCalibrator, CFGNNHyperConfig
from .ncpnet import NCPNETStyleCalibrator, NCPNETHyperConfig

__all__ = [
    "ACI",
    "ACIConfig",
    "SplitCP",
    "SplitCPConfig",
    "ConformalPID",
    "ConformalPIDConfig",
    "AgACI",
    "AgACIConfig",
    "CFGNNSmoothedCalibrator",
    "CFGNNHyperConfig",
    "NCPNETStyleCalibrator",
    "NCPNETHyperConfig",
]
