"""Unit tests for inference service."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.schemas.request import InferCovGroup
from app.services.inference_service import (
    InferenceError,
    _resolve_target_model_dir,
    _to_float_list,
    run_inference,
)


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


def test_resolve_target_model_dir_success(tmp_path: Path):
    release_dir = tmp_path / "release"
    target_dir = release_dir / "finetuned-ckpt_value1"
    target_dir.mkdir(parents=True)

    resolved = _resolve_target_model_dir(release_dir, "value1")
    assert resolved == target_dir


def test_resolve_target_model_dir_not_found(tmp_path: Path):
    with pytest.raises(InferenceError) as exc:
        _resolve_target_model_dir(tmp_path, "missing")
    assert exc.value.code == 404
    assert exc.value.message == "model for target 'missing' not found"


def test_to_float_list_from_tensor_like():
    vals = _to_float_list(_FakeTensor([1, 2.5, 3]))
    assert vals == [1.0, 2.5, 3.0]


def test_to_float_list_from_plain_list():
    vals = _to_float_list([1, 2, 3])
    assert vals == [1.0, 2.0, 3.0]


def test_run_inference_model_path_not_found(tmp_path: Path):
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("value1,value2\n1,2\n3,4\n", encoding="utf-8")
    group = [InferCovGroup(target="value1", covariates=["value2"])]

    with pytest.raises(InferenceError) as exc:
        run_inference(
            model_path=str(tmp_path / "no_model"),
            cov_group=group,
            prediction_length=2,
            context_length=2,
            csv_path=str(csv_path),
        )
    assert exc.value.code == 404
    assert exc.value.message == "model path not found"


def test_run_inference_csv_not_found(tmp_path: Path):
    model_root = tmp_path / "release"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True)
    group = [InferCovGroup(target="value1", covariates=["value2"])]

    with pytest.raises(InferenceError) as exc:
        run_inference(
            model_path=str(model_root),
            cov_group=group,
            prediction_length=2,
            context_length=2,
            csv_path=str(tmp_path / "no.csv"),
        )
    assert exc.value.code == 404
    assert exc.value.message == "csv_path not found"


def test_run_inference_history_too_short(tmp_path: Path):
    model_root = tmp_path / "release"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True)
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("value1,value2\n1,2\n", encoding="utf-8")
    group = [InferCovGroup(target="value1", covariates=["value2"])]

    with pytest.raises(InferenceError) as exc:
        run_inference(
            model_path=str(model_root),
            cov_group=group,
            prediction_length=2,
            context_length=2,
            csv_path=str(csv_path),
        )
    assert exc.value.code == 400
    assert exc.value.message == "history length is insufficient"


def test_run_inference_success_with_mock_pipeline(tmp_path: Path):
    model_root = tmp_path / "release"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True)
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("value1,value2\n1,2\n3,4\n5,6\n", encoding="utf-8")
    group = [InferCovGroup(target="value1", covariates=["value2"])]

    with patch("app.services.inference_service.load_local_model", return_value=_FakePipeline([0.1, 0.2])):
        result = run_inference(
            model_path=str(model_root),
            cov_group=group,
            prediction_length=2,
            context_length=2,
            csv_path=str(csv_path),
        )

    assert len(result) == 1
    assert result[0].target == "value1"
    assert result[0].prediction == [0.1]
