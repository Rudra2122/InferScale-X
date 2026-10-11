"""Prometheus metrics, structured JSON logging and optional OpenTelemetry tracing."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)

REQUESTS = Counter("inferscale_requests_total", "Requests by outcome",
                   ["tenant", "backend", "status"], registry=REGISTRY)
REJECTS = Counter("inferscale_rejections_total", "Admission rejections",
                  ["tenant", "reason"], registry=REGISTRY)
E2E = Histogram("inferscale_request_seconds", "End-to-end latency at gateway",
                ["tenant", "stream"], buckets=LAT_BUCKETS, registry=REGISTRY)
TTFT = Histogram("inferscale_ttft_seconds", "Time to first streamed chunk at gateway",
                 ["tenant"], buckets=LAT_BUCKETS, registry=REGISTRY)
OUT_TOKENS = Counter("inferscale_output_tokens_total", "Output tokens (from backend usage)",
                     ["tenant"], registry=REGISTRY)
INFLIGHT = Gauge("inferscale_inflight", "In-flight requests", registry=REGISTRY)
BACKEND_HEALTHY = Gauge("inferscale_backend_healthy", "1 if backend healthy", ["backend"], registry=REGISTRY)
BACKEND_OUTSTANDING = Gauge("inferscale_backend_outstanding", "Outstanding requests per backend",
                            ["backend"], registry=REGISTRY)
RETRIES = Counter("inferscale_retries_total", "Retries after backend failure", ["backend"], registry=REGISTRY)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(time.time(), 6), "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage()}
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str = "inferscale") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(os.environ.get("INFERSCALE_LOG_LEVEL", "INFO"))
        log.propagate = False
    return log


def log_event(log: logging.Logger, msg: str, **fields) -> None:
    log.info(msg, extra={"fields": fields})


# ---------------- tracing (optional) ----------------
_tracer = None


def init_tracing(service_name: str = "inferscale-gateway"):
    """Enable OTel if the SDK is installed and OTEL_EXPORTER is set ('console' or 'otlp')."""
    global _tracer
    exporter = os.environ.get("OTEL_EXPORTER")
    if not exporter:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
    except Exception as e:  # tracing must never break serving
        get_logger().warning(f"tracing disabled: {e}")
        _tracer = None
    return _tracer


@contextlib.contextmanager
def span(name: str, **attrs):
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s
