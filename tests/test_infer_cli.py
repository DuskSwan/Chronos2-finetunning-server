import json
from pathlib import Path
from unittest.mock import patch

from app.cli.infer_cli import main


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.asarray(self._values, dtype=float)


class _FakePipeline:
    def __init__(self, values):
        self._values = values

    def predict(self, inputs, prediction_length):
        _ = inputs
        _ = prediction_length
        return [_FakeTensor(self._values)]


def test_cli_infer_success_with_metadata(tmp_path: Path):
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("value1,value2\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")

    model_root = tmp_path / "model"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True, exist_ok=True)
    (model_root / "metadata.json").write_text(
        json.dumps(
            {
                "selected_groups": [{"target": "value1", "covariates": ["value2"]}],
                "prediction_length": 2,
                "context_length": 2,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "result" / "output.json"

    with patch("app.services.inference_service.load_local_model", return_value=_FakePipeline([0.1, 0.2])):
        rc = main(
            [
                "--model-path",
                str(model_root),
                "--csv-path",
                str(csv_path),
                "--output-path",
                str(output_path),
            ]
        )

    assert rc == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["code"] == 0
    assert payload["message"] == "success"
    assert payload["data"]["model_path"] == str(model_root)
    assert payload["data"]["csv_path"] == str(csv_path)
    assert payload["data"]["predictions"][0]["target"] == "value1"


def test_cli_infer_missing_metadata_and_missing_params(tmp_path: Path):
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("value1,value2\n1,2\n3,4\n5,6\n", encoding="utf-8")

    model_root = tmp_path / "model"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True, exist_ok=True)

    output_path = tmp_path / "result" / "output.json"
    rc = main(
        [
            "--model-path",
            str(model_root),
            "--csv-path",
            str(csv_path),
            "--output-path",
            str(output_path),
        ]
    )

    assert rc == 4
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["code"] == 400
    assert payload["message"] == "cov_group is required when metadata.json is missing"


def test_cli_infer_targets_filter(tmp_path: Path):
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("value1,value2\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")

    model_root = tmp_path / "model"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True, exist_ok=True)
    (model_root / "finetuned-ckpt_value2").mkdir(parents=True, exist_ok=True)
    (model_root / "metadata.json").write_text(
        json.dumps(
            {
                "selected_groups": [
                    {"target": "value1", "covariates": ["value2"]},
                    {"target": "value2", "covariates": ["value1"]},
                ],
                "prediction_length": 2,
                "context_length": 2,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "result" / "output.json"

    def fake_loader(path, device="cpu"):
        _ = device
        if Path(path).name.endswith("value1"):
            return _FakePipeline([0.1, 0.2])
        return _FakePipeline([0.3, 0.4])

    with patch("app.services.inference_service.load_local_model", side_effect=fake_loader):
        rc = main(
            [
                "--model-path",
                str(model_root),
                "--csv-path",
                str(csv_path),
                "--output-path",
                str(output_path),
                "--targets",
                "value2",
            ]
        )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    preds = payload["data"]["predictions"]
    assert len(preds) == 1
    assert preds[0]["target"] == "value2"


def test_cli_infer_targets_not_found(tmp_path: Path):
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("value1,value2\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")

    model_root = tmp_path / "model"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True, exist_ok=True)
    (model_root / "metadata.json").write_text(
        json.dumps(
            {
                "selected_groups": [{"target": "value1", "covariates": ["value2"]}],
                "prediction_length": 2,
                "context_length": 2,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "result" / "output.json"

    rc = main(
        [
            "--model-path",
            str(model_root),
            "--csv-path",
            str(csv_path),
            "--output-path",
            str(output_path),
            "--targets",
            "value_x",
        ]
    )

    assert rc == 4
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["code"] == 400
    assert "targets not found in inference groups" in payload["message"]
