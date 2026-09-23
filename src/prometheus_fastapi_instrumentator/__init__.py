from .cardinality import CardinalityEvent, LabelCardinalityBudget
from .instrumentation import PrometheusFastApiInstrumentator

__version__ = "8.1.0"

Instrumentator = PrometheusFastApiInstrumentator

__all__ = [
    "CardinalityEvent",
    "Instrumentator",
    "LabelCardinalityBudget",
    "PrometheusFastApiInstrumentator",
]
