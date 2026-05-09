"""ZMQ REQ/REP inference server for ExternalNode integration."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import zmq

from app.services.inference_service import InferenceError, run_inference_from_dataframe


def build_success_response(predictions: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "code": 200,
        "type": "timeseries",
        "version": "1.0",
        "data": predictions,
        "message": "success",
    }


def build_error_response(code: int, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "type": "timeseries",
        "version": "1.0",
        "data": {},
        "message": message,
    }


def _parse_request_payload(raw: str) -> pd.DataFrame:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InferenceError(400, f"invalid json payload: {exc.msg}") from exc

    if not isinstance(payload, list):
        raise InferenceError(400, "payload must be a list of objects")
    if not payload:
        raise InferenceError(400, "payload list cannot be empty")
    if any(not isinstance(item, dict) for item in payload):
        raise InferenceError(400, "payload items must be objects")

    return pd.DataFrame(payload)


def process_request(model_path: str, raw_request: str) -> dict[str, Any]:
    try:
        df = _parse_request_payload(raw_request)
        prediction_items = run_inference_from_dataframe(
            model_path=model_path,
            cov_group=None,
            prediction_length=None,
            context_length=None,
            dataframe=df,
            require_metadata=True,
        )
        result = {item.target: item.prediction for item in prediction_items}
        return build_success_response(result)
    except InferenceError as exc:
        return build_error_response(exc.code, exc.message)
    except Exception as exc:
        return build_error_response(500, f"internal error: {exc}")


def serve_req_rep(model_path: str, endpoint: str) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(endpoint)

    try:
        while True:
            raw = socket.recv_string()
            response = process_request(model_path=model_path, raw_request=raw)
            socket.send_string(json.dumps(response, ensure_ascii=False))
    finally:
        socket.close(0)
        context.term()
