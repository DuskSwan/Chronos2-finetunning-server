from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.schemas.request import InferCovGroup
from app.services.inference_service import InferenceError, run_inference_from_dataframe


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


def test_run_inference_from_dataframe_success(tmp_path: Path):
    model_root = tmp_path / "release"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True)
    df = pd.DataFrame({"value1": [1, 3, 5], "value2": [2, 4, 6]})
    group = [InferCovGroup(target="value1", covariates=["value2"])]

    with patch("app.services.inference_service.load_local_model", return_value=_FakePipeline([0.1, 0.2])):
        result = run_inference_from_dataframe(
            model_path=str(model_root),
            cov_group=group,
            prediction_length=2,
            context_length=2,
            dataframe=df,
        )

    assert len(result) == 1
    assert result[0].target == "value1"
    assert result[0].prediction == [0.1]


def test_run_inference_from_dataframe_require_metadata(tmp_path: Path):
    model_root = tmp_path / "release"
    (model_root / "finetuned-ckpt_value1").mkdir(parents=True)
    df = pd.DataFrame({"value1": [1, 3, 5], "value2": [2, 4, 6]})

    with pytest.raises(InferenceError) as exc:
        run_inference_from_dataframe(
            model_path=str(model_root),
            cov_group=None,
            prediction_length=None,
            context_length=None,
            dataframe=df,
            require_metadata=True,
        )
    assert exc.value.code == 400
    assert exc.value.message == "metadata.json is required"
