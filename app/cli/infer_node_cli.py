"""ExternalNode inference binary entrypoint."""

from __future__ import annotations

import argparse

from app.runtime.zmq_infer_server import serve_req_rep


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronos_infer_node",
        description="Run ZMQ inference node for ExternalNode integration.",
    )
    parser.add_argument("--model-path", required=True, help="Released model directory path")
    parser.add_argument("--zmq-endpoint", required=True, help="ZMQ bind endpoint, e.g. tcp://127.0.0.1:52345")
    parser.add_argument("--zmq-protocol", required=True, help="ZMQ protocol from platform (REQ/DEALER)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    protocol = str(args.zmq_protocol).strip().upper()
    if protocol != "REQ":
        print(f"unsupported zmq protocol: {protocol}; only REQ is supported")
        return 2

    serve_req_rep(model_path=args.model_path, endpoint=args.zmq_endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
