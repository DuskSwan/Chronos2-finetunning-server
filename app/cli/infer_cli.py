"""Offline inference CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from app.services.inference_service import InferenceError, run_inference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronos_infer",
        description="Run offline Chronos inference from released model directory.",
    )
    parser.add_argument("--model-path", required=True, help="Released model directory path")
    parser.add_argument("--csv-path", required=True, help="Input CSV file path")
    parser.add_argument("--output-path", required=True, help="Output JSON file path")
    parser.add_argument("--prediction-length", type=int, default=None, help="Override prediction_length")
    parser.add_argument("--context-length", type=int, default=None, help="Override context_length")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated targets to infer, e.g. value1,value2",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed error traceback")
    parser.add_argument("--version", action="version", version="chronos_infer 0.1.0")
    return parser


def _parse_targets(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",")]
    items = [item for item in items if item]
    if not items:
        raise ValueError("--targets cannot be empty")
    unique = list(dict.fromkeys(items))
    return unique


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        targets = _parse_targets(args.targets)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}")
        return 2

    output_path = Path(args.output_path)

    try:
        predictions = run_inference(
            model_path=args.model_path,
            cov_group=None,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            csv_path=args.csv_path,
            target_filter=targets,
        )
        payload = {
            "code": 0,
            "message": "success",
            "data": {
                "model_path": args.model_path,
                "csv_path": args.csv_path,
                "predictions": [item.model_dump() for item in predictions],
            },
        }
        _write_json(output_path, payload)
        print(f"success: output written to {output_path}")
        return 0
    except InferenceError as exc:
        payload = {"code": exc.code, "message": exc.message, "data": None}
        _write_json(output_path, payload)
        print(exc.message)
        if exc.code == 404:
            return 3
        return 4
    except Exception as exc:
        payload = {"code": 500, "message": "unexpected error", "data": None}
        _write_json(output_path, payload)
        print(f"unexpected error: {exc}")
        if args.verbose:
            print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
