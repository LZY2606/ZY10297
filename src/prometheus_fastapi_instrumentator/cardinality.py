"""Label cardinality budgeting for metric families.

This module implements an opt-in gate that bounds the number of distinct
label value combinations (time series) a single metric family can produce
within one registry. It protects against unbounded cardinality caused by
untemplated paths, unexpected status values or user-controlled label values.

The budget is configured once via `LabelCardinalityBudget` and handed to the
metric functions in `metrics` (or to the instrumentator, which forwards it to
the default metrics). Every metric family created with the budget gets its
own independent gate. The default behavior of the library is unchanged: if no
budget is configured, cardinality is unlimited.

Multi-worker / multiprocess note: the gate state lives in the current
process and is scoped to the registry the metric family is registered with.
It is not shared across worker processes.
"""

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

STRATEGY_DROP = "drop"
STRATEGY_OVERFLOW = "overflow"
STRATEGY_OBSERVER = "observer"

_STRATEGIES = (STRATEGY_DROP, STRATEGY_OVERFLOW, STRATEGY_OBSERVER)


@dataclass(frozen=True)
class CardinalityEvent:
    """Structured notification emitted when a label tuple is rejected.

    Deliberately contains no label values so that high-cardinality (and
    potentially sensitive) strings are never written to logs by observers.
    """

    family: str
    label_names: Tuple[str, ...]
    reason: str
    strategy: str
    dropped: int
    overflowed: int


class LabelCardinalityGate:
    """Tracks confirmed label tuples for a single metric family.

    One gate exists per metric family per registry. All budget decisions are
    made here, before the actual `observe`/`inc` call on the metric.
    """

    def __init__(
        self,
        family: str,
        budget: "LabelCardinalityBudget",
        label_names: Sequence[str],
    ) -> None:
        self.family = family
        self._budget = budget
        self._label_names = tuple(label_names)
        self._lock = threading.Lock()
        self._confirmed: set = set()
        self._label_values: Dict[str, set] = {
            name: set()
            for name in self._label_names
            if name in budget.max_label_cardinality
        }
        self._admitted = 0
        self._dropped = 0
        self._overflowed = 0

    def reset(self) -> None:
        """Forgets all confirmed tuples and counters.

        The internal mutable sets are replaced with new ones instead of being
        cleared in place, so nothing holding on to the old state (for example
        a stale registry) can observe or influence the fresh state.
        """

        with self._lock:
            self._confirmed = set()
            self._label_values = {name: set() for name in self._label_values}
            self._admitted = 0
            self._dropped = 0
            self._overflowed = 0

    def check(self, label_values: Sequence[str]) -> Optional[Tuple[str, ...]]:
        """Decides what to do with a label tuple before observing.

        Args:
            label_values: The label values in the order of the label names
                this gate was created with.

        Returns:
            The tuple of label values to use for the observation, or `None`
            if the observation must be dropped. Overflowing tuples are mapped
            to the fixed overflow identity.
        """

        budget = self._budget
        key = tuple(label_values)
        event: Optional[CardinalityEvent] = None

        with self._lock:
            if key in self._confirmed:
                self._admitted += 1
                return key

            reason = self._violation(key)
            if reason is None:
                self._confirmed.add(key)
                for name, value in zip(self._label_names, key):
                    if name in self._label_values:
                        self._label_values[name].add(value)
                self._admitted += 1
                return key

            if budget.strategy == STRATEGY_OVERFLOW:
                self._overflowed += 1
                result: Optional[Tuple[str, ...]] = self._overflow_key()
                # The overflow tuple itself occupies exactly one identity.
                self._confirmed.add(result)
            else:
                # STRATEGY_DROP and STRATEGY_OBSERVER both skip the
                # observation. STRATEGY_OBSERVER only differs in that an
                # observer is mandatory.
                self._dropped += 1
                result = None

            if budget.observer is not None:
                event = CardinalityEvent(
                    family=self.family,
                    label_names=self._label_names,
                    reason=reason,
                    strategy=budget.strategy,
                    dropped=self._dropped,
                    overflowed=self._overflowed,
                )

        # Notify outside the lock so observers can safely re-enter.
        if event is not None and budget.observer is not None:
            budget.observer(event)

        return result

    def _overflow_key(self) -> Tuple[str, ...]:
        return tuple(self._budget.overflow_value for _ in self._label_names)

    def _violation(self, key: Tuple[str, ...]) -> Optional[str]:
        """Returns the reason why the tuple would exceed the budget, if any."""

        budget = self._budget
        if (
            budget.max_cardinality is not None
            and len(self._confirmed) >= budget.max_cardinality
        ):
            return "max_cardinality"
        for name, value in zip(self._label_names, key):
            if name not in self._label_values:
                continue
            seen = self._label_values[name]
            if value not in seen and len(seen) >= budget.max_label_cardinality[name]:
                return f"label:{name}"
        return None

    def stats(self) -> Dict[str, int]:
        """Structured counters for this family. Contains no label values."""

        with self._lock:
            return {
                "admitted": self._admitted,
                "dropped": self._dropped,
                "overflowed": self._overflowed,
                "cardinality": len(self._confirmed),
            }


class LabelCardinalityBudget:
    """Configuration for per-metric-family label cardinality limits.

    Args:
        max_cardinality: Maximum number of distinct label tuples per metric
            family. `None` means unlimited.
        max_label_cardinality: Mapping of label name to maximum number of
            distinct values for that label within one metric family.
        strategy: What to do when a new tuple would exceed the budget.
            `"drop"` skips the observation, `"overflow"` maps it to the fixed
            `overflow_value` for every label (a single shared identity), and
            `"observer"` skips the observation and notifies the `observer`.
            Defaults to `"overflow"`.
        overflow_value: Fixed label value used for the overflow identity.
            Defaults to `"__overflow__"`.
        observer: Optional callable invoked with a `CardinalityEvent` whenever
            a tuple is rejected. The event never contains label values.
    """

    def __init__(
        self,
        max_cardinality: Optional[int] = None,
        max_label_cardinality: Optional[Dict[str, int]] = None,
        strategy: str = STRATEGY_OVERFLOW,
        overflow_value: str = "__overflow__",
        observer: Optional[Callable[[CardinalityEvent], None]] = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {_STRATEGIES}."
            )
        if max_cardinality is not None and max_cardinality < 1:
            raise ValueError("max_cardinality must be a positive integer.")
        for name, limit in (max_label_cardinality or {}).items():
            if limit < 1:
                raise ValueError(f"Limit for label '{name}' must be a positive integer.")
        if not overflow_value:
            raise ValueError("overflow_value must be a non-empty string.")
        if strategy == STRATEGY_OBSERVER and observer is None:
            raise ValueError("strategy 'observer' requires an observer callable.")

        self.max_cardinality = max_cardinality
        self.max_label_cardinality = dict(max_label_cardinality or {})
        self.strategy = strategy
        self.overflow_value = overflow_value
        self.observer = observer

        self._lock = threading.Lock()
        self._gates: List[LabelCardinalityGate] = []

    def gate_for(self, family: str, label_names: Sequence[str]) -> LabelCardinalityGate:
        """Creates a new gate for a metric family.

        Called by the metric functions once per created metric. Because every
        metric family (per registry) gets its own gate, replacing the registry
        never reuses old mutable state.
        """

        gate = LabelCardinalityGate(family, self, label_names)
        with self._lock:
            self._gates.append(gate)
        return gate

    def reset(self) -> None:
        """Resets all gates created from this budget."""

        with self._lock:
            gates = list(self._gates)
        for gate in gates:
            gate.reset()

    def stats(self) -> Dict[str, Dict[str, int]]:
        """Aggregated structured counters per family. No label values."""

        with self._lock:
            gates = list(self._gates)
        aggregated: Dict[str, Dict[str, int]] = {}
        for gate in gates:
            stats = gate.stats()
            family_stats = aggregated.setdefault(
                gate.family,
                {"admitted": 0, "dropped": 0, "overflowed": 0, "cardinality": 0},
            )
            for key, value in stats.items():
                family_stats[key] += value
        return aggregated
