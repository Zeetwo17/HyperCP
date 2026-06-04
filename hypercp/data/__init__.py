"""Data pipelines for HyperCP.

Modules
-------
- supplygraph         : SupplyGraph (real, Wasi 2024) hypergraph reformulation
- supplysim           : Chang et al. AAAI 2025 simulator wrapper
- supplygraph_stressed: SupplyGraph topology + simulator shocks (ours)
- m5                  : M5 hierarchy -> hyperedge encoding (DONE)
- scr                 : SCR + SCR-NR + SCR-ER variants (in supplysim.py for now)
- splits              : rolling-origin temporal splits (in supplygraph / m5)
"""

from .supplygraph import (
    SupplyGraphHypergraph,
    Hyperedge,
    PartitionClass,
    QAReport,
)
from .m5 import M5Hypergraph, M5QAReport

__all__ = [
    "SupplyGraphHypergraph",
    "Hyperedge",
    "PartitionClass",
    "QAReport",
    "M5Hypergraph",
    "M5QAReport",
]
