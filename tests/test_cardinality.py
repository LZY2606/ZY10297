"""Tests for the per-metric-family label cardinality budget."""

import threading
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from prometheus_client import CollectorRegistry, generate_latest
from starlette.testclient import TestClient

from prometheus_fastapi_instrumentator import (
    Instrumentator,
    LabelCardinalityBudget,
    metrics,
)
from prometheus_fastapi_instrumentator.cardinality import CardinalityEvent

# ------------------------------------------------------------------------------
# Setup


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/items/{item_id}")
    def read_item(item_id: int):
        return {"item_id": item_id}

    @app.get("/ignore")
    def read_ignore():
        return "Should be ignored"

    @app.get("/runtime_error")
    def always_error():
        raise RuntimeError()

    @app.get("/stream")
    def stream():
        def gen():
            yield b"chunk1"
            yield b"chunk2"

        return StreamingResponse(gen())

    return app


def instrument(
    app: FastAPI,
    registry: CollectorRegistry,
    budget: Optional[LabelCardinalityBudget],
    **kwargs,
) -> None:
    """Instruments with raw paths as handlers for untemplated routes."""

    Instrumentator(
        should_group_untemplated=False,
        registry=registry,
        label_cardinality_budget=budget,
        **kwargs,
    ).instrument(app)


def exposition(registry: CollectorRegistry) -> str:
    return generate_latest(registry).decode()


def series_count(body: str, metric_name: str) -> int:
    return sum(
        1
        for line in body.splitlines()
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " ")
    )


# ------------------------------------------------------------------------------
# Gate unit tests (concurrency)


def test_gate_concurrent_same_tuple_single_slot():
    budget = LabelCardinalityBudget(max_cardinality=10)
    gate = budget.gate_for("fam", ("handler", "method"))
    barrier = threading.Barrier(8)
    results: List[Optional[tuple]] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        result = gate.check(("/items/1", "GET"))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(result == ("/items/1", "GET") for result in results)
    assert gate.stats()["cardinality"] == 1
    assert gate.stats()["admitted"] == 8


def test_gate_concurrent_distinct_tuples_respect_budget():
    budget = LabelCardinalityBudget(max_cardinality=4, strategy="overflow")
    gate = budget.gate_for("fam", ("handler",))
    barrier = threading.Barrier(16)
    results: List[tuple] = []
    lock = threading.Lock()

    def worker(num: int):
        barrier.wait()
        result = gate.check((f"/notfound/{num}",))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker, args=(num,)) for num in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stats = gate.stats()
    # 4 regular identities plus exactly one shared overflow identity.
    assert stats["cardinality"] == 5
    assert stats["admitted"] + stats["overflowed"] == 16
    assert stats["overflowed"] > 0
    assert (
        sum(1 for result in results if result == ("__overflow__",)) == stats["overflowed"]
    )


# ------------------------------------------------------------------------------
# HTTP level tests


def test_exactly_at_limit_then_overflow():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=2)
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")
    client.get("/notfound/3")
    client.get("/notfound/1")

    body = exposition(registry)
    assert (
        'http_requests_total{handler="/notfound/1",method="GET",status="4xx"} 2.0' in body
    )
    assert (
        'http_requests_total{handler="/notfound/2",method="GET",status="4xx"} 1.0' in body
    )
    assert 'handler="/notfound/3"' not in body
    assert (
        'http_requests_total{handler="__overflow__",method="__overflow__",'
        'status="__overflow__"} 1.0' in body
    )
    assert series_count(body, "http_requests_total") == 3


def test_drop_strategy():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1, strategy="drop")
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")

    body = exposition(registry)
    assert (
        'http_requests_total{handler="/notfound/1",method="GET",status="4xx"} 1.0' in body
    )
    assert "__overflow__" not in body
    assert series_count(body, "http_requests_total") == 1
    assert budget.stats()["http_requests_total"]["dropped"] == 1


def test_observer_strategy_event_has_no_label_values():
    events: List[CardinalityEvent] = []
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(
        max_cardinality=1, strategy="observer", observer=events.append
    )
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")
    client.get("/notfound/3")

    total_events = [e for e in events if e.family == "http_requests_total"]
    assert len(total_events) == 2
    # Every labeled family of the default instrumentation is gated.
    assert {e.family for e in events} == {
        "http_requests_total",
        "http_request_size_bytes",
        "http_response_size_bytes",
        "http_request_duration_seconds",
    }
    event = total_events[0]
    assert event.family == "http_requests_total"
    assert event.label_names == ("method", "status", "handler")
    assert event.reason == "max_cardinality"
    assert event.strategy == "observer"
    assert event.dropped == 1
    # The event (and thus anything logged from it) must not contain label values.
    assert "/notfound/2" not in str(event)
    assert "/notfound/3" not in str(event)
    assert "/notfound/2" not in repr(events)
    assert series_count(exposition(registry), "http_requests_total") == 1


def test_per_label_distinct_limit():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_label_cardinality={"handler": 1})
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")

    body = exposition(registry)
    assert (
        'http_requests_total{handler="/notfound/1",method="GET",status="4xx"} 1.0' in body
    )
    assert (
        'http_requests_total{handler="__overflow__",method="__overflow__",'
        'status="__overflow__"} 1.0' in body
    )
    assert budget.stats()["http_requests_total"]["overflowed"] == 1


def test_reset_restores_budget():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")
    body = exposition(registry)
    assert 'handler="/notfound/2"' not in body

    budget.reset()
    client.get("/notfound/2")
    body = exposition(registry)
    assert (
        'http_requests_total{handler="/notfound/2",method="GET",status="4xx"} 1.0' in body
    )


def test_reset_replaces_mutable_state():
    budget = LabelCardinalityBudget(max_cardinality=1)
    gate = budget.gate_for("fam", ("handler",))
    gate.check(("/notfound/1",))
    old_confirmed = gate._confirmed
    old_label_values = gate._label_values

    budget.reset()

    assert gate._confirmed is not old_confirmed
    assert gate._label_values is not old_label_values
    assert old_confirmed == {("/notfound/1",)}
    assert gate.stats()["cardinality"] == 0
    assert gate.check(("/notfound/2",)) == ("/notfound/2",)


def test_two_registries_are_isolated():
    budget = LabelCardinalityBudget(max_cardinality=1)
    registry_one = CollectorRegistry()
    registry_two = CollectorRegistry()

    app_one = create_app()
    instrument(app_one, registry_one, budget)
    app_two = create_app()
    instrument(app_two, registry_two, budget)

    client_one = TestClient(app_one)
    client_two = TestClient(app_two)

    client_one.get("/notfound/1")
    client_one.get("/notfound/2")
    client_two.get("/notfound/2")

    body_one = exposition(registry_one)
    body_two = exposition(registry_two)
    assert 'handler="/notfound/2"' not in body_one
    assert (
        'http_requests_total{handler="/notfound/2",method="GET",status="4xx"} 1.0'
        in body_two
    )


def test_http_concurrent_same_and_different_labels():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=100)
    app = create_app()
    instrument(app, registry, budget)

    # Force middleware stack construction before threads start.
    TestClient(app).get("/items/0")

    barrier = threading.Barrier(8)

    def worker(num: int):
        client = TestClient(app)
        barrier.wait()
        # All threads share the templated tuple, each adds a distinct one.
        client.get(f"/items/{num}")
        client.get(f"/notfound/{num}")

    threads = [threading.Thread(target=worker, args=(num,)) for num in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    body = exposition(registry)
    assert (
        'http_requests_total{handler="/items/{item_id}",method="GET",status="2xx"}'
        " 9.0" in body
    )
    # One templated tuple plus 8 distinct untemplated tuples.
    assert series_count(body, "http_requests_total") == 9


def test_exception_handler_goes_through_gate():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app, raise_server_exceptions=False)

    client.get("/items/1")
    response = client.get("/runtime_error")
    assert response.status_code == 500

    body = exposition(registry)
    assert 'handler="/runtime_error"' not in body
    assert (
        'http_requests_total{handler="__overflow__",method="__overflow__",'
        'status="__overflow__"} 1.0' in body
    )


def test_streaming_response_goes_through_gate():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/items/1")
    response = client.get("/stream")
    assert response.status_code == 200
    assert response.content == b"chunk1chunk2"

    body = exposition(registry)
    assert 'handler="/stream"' not in body
    assert (
        'http_requests_total{handler="__overflow__",method="__overflow__",'
        'status="__overflow__"} 1.0' in body
    )


def test_excluded_handler_consumes_no_budget():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    instrument(app, registry, budget, excluded_handlers=["/ignore"])
    client = TestClient(app)

    client.get("/ignore")
    client.get("/ignore")
    client.get("/items/1")

    body = exposition(registry)
    assert "/ignore" not in body
    assert (
        'http_requests_total{handler="/items/{item_id}",method="GET",status="2xx"}'
        " 1.0" in body
    )
    assert budget.stats()["http_requests_total"]["cardinality"] == 1


def test_untemplated_routes_grouped_into_single_slot():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    # Default: untemplated routes are grouped into handler "none".
    Instrumentator(registry=registry, label_cardinality_budget=budget).instrument(app)
    client = TestClient(app)

    client.get("/does/not/exist/1")
    client.get("/does/not/exist/2")
    client.get("/does/not/exist/3")

    body = exposition(registry)
    assert 'http_requests_total{handler="none",method="GET",status="4xx"} 3.0' in body
    assert budget.stats()["http_requests_total"]["cardinality"] == 1


def test_custom_labels_go_through_gate():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    Instrumentator(should_group_untemplated=False, registry=registry).add(
        metrics.default(
            registry=registry,
            custom_labels={"env": "test"},
            label_cardinality_budget=budget,
        )
    ).instrument(app)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")

    body = exposition(registry)
    assert (
        'http_requests_total{env="test",handler="/notfound/1",method="GET",'
        'status="4xx"} 1.0' in body
    )
    assert (
        'http_requests_total{env="__overflow__",handler="__overflow__",'
        'method="__overflow__",status="__overflow__"} 1.0' in body
    )


def test_default_is_unlimited_without_budget():
    registry = CollectorRegistry()
    app = create_app()
    instrument(app, registry, None)
    client = TestClient(app)

    for num in range(5):
        client.get(f"/notfound/{num}")

    body = exposition(registry)
    assert series_count(body, "http_requests_total") == 5
    assert "__overflow__" not in body


def test_stats_are_structured_and_free_of_label_values():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=1)
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/notfound/1")
    client.get("/notfound/2")

    stats = budget.stats()
    family_stats = stats["http_requests_total"]
    assert family_stats == {
        "admitted": 1,
        "dropped": 0,
        "overflowed": 1,
        "cardinality": 2,
    }
    assert "/notfound/1" not in repr(stats)
    assert "/notfound/2" not in repr(stats)


def test_metric_names_stay_compatible():
    registry = CollectorRegistry()
    budget = LabelCardinalityBudget(max_cardinality=100)
    app = create_app()
    instrument(app, registry, budget)
    client = TestClient(app)

    client.get("/items/1")

    body = exposition(registry)
    assert "# HELP http_requests_total" in body
    assert "# HELP http_request_size_bytes" in body
    assert "# HELP http_response_size_bytes" in body
    assert "# HELP http_request_duration_highr_seconds" in body
    assert "# HELP http_request_duration_seconds" in body


def test_invalid_budget_configuration():
    for kwargs in (
        {"strategy": "bogus"},
        {"strategy": "observer"},
        {"max_cardinality": 0},
        {"max_label_cardinality": {"handler": 0}},
        {"overflow_value": ""},
    ):
        try:
            LabelCardinalityBudget(**kwargs)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
