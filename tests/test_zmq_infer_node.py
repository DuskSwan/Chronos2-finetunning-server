import json

from app.cli.infer_node_cli import main as node_main
from app.runtime.zmq_infer_server import process_request


def test_node_cli_reject_dealer():
    rc = node_main(
        [
            "--model-path",
            "D:/x/model",
            "--zmq-endpoint",
            "tcp://127.0.0.1:52345",
            "--zmq-protocol",
            "DEALER",
        ]
    )
    assert rc == 2


def test_process_request_invalid_json():
    resp = process_request(model_path="D:/x/model", raw_request="not json")
    assert resp["code"] == 400
    assert resp["data"] == {}


def test_process_request_invalid_payload_shape():
    resp = process_request(model_path="D:/x/model", raw_request=json.dumps({"a": 1}))
    assert resp["code"] == 400
    assert "list" in resp["message"]
