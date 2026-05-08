#!/usr/bin/env python
"""End-to-end demo: fine-tune + publish + infer using mock_train_data.csv."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from app.core.config import get_settings

PORT = get_settings().port
BASE_URL = f"http://127.0.0.1:{PORT}"
DATA_FILE = Path("mock_train_data.csv").resolve()
SELECTED_GROUPS = [
    {
        "target": "value1",
        "covariates": ["value2", "value3", "value4"],
    }
]
PUBLISH_USER_ID = 10001
PUBLISH_VERSION = "1.0.0"


def _print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _require_data_file() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {DATA_FILE}. 请确认 mock_train_data.csv 在仓库根目录。"
        )


def _api_headers() -> dict[str, str]:
    """构造兼容接口请求头（需要时自动带 Bearer Token）。"""
    token = get_settings().api_bearer_token
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _poll_job(job_id: str, timeout_sec: int = 300) -> dict:
    """轮询任务状态直到完成或失败。"""
    start = time.time()
    while True:
        resp = requests.get(f"{BASE_URL}/v1/finetune/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        print(f"Job {job_id} status: {status}")

        if status in {"completed", "failed", "cancelled"}:
            return data

        if time.time() - start > timeout_sec:
            raise TimeoutError(f"等待任务超时: {job_id}")

        time.sleep(2)


def _publish_model(job_id: str) -> str:
    payload = {
        "user_id": PUBLISH_USER_ID,
        "version": PUBLISH_VERSION,
        "job_id": job_id,
    }
    print("\n[4] Publish Model")
    print("POST /api/model/publish")
    print(json.dumps(payload, indent=2))
    resp = requests.post(
        f"{BASE_URL}/api/model/publish",
        headers=_api_headers(),
        json=payload,
    )
    print(f"Status: {resp.status_code}")
    body = resp.json()
    print(json.dumps(body, indent=2, ensure_ascii=False))
    if body.get("code") != 0 or not body.get("data"):
        raise RuntimeError(f"发布模型失败: {body}")
    return body["data"]["model_path"]


def _infer_model(model_path: str) -> dict:
    payload = {
        "model_path": model_path,
        "cov_group": SELECTED_GROUPS,
        "prediction_length": 8,
        "context_length": 32,
        "csv_path": str(DATA_FILE),
    }
    print("\n[5] Infer with Published Model")
    print("POST /api/model/infer")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    resp = requests.post(
        f"{BASE_URL}/api/model/infer",
        headers=_api_headers(),
        json=payload,
    )
    print(f"Status: {resp.status_code}")
    body = resp.json()
    print(json.dumps(body, indent=2, ensure_ascii=False))
    if body.get("code") != 0:
        raise RuntimeError(f"模型推理失败: {body}")
    return body


def demo() -> None:
    _require_data_file()

    _print_section("CHRONOS-2 FINE-TUNING JOB SERVICE - DEMO")

    # 1. Health Check
    print("[1] Health Check")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"GET /health -> {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    # 2. Create Job (mock_train_data.csv + selected_groups)
    print("\n[2] Create Fine-tuning Job")
    payload = {
        "train_data_path": str(DATA_FILE),
        "prediction_length": 64,
        "context_length": 512,
        "finetune_mode": "lora",
        "learning_rate": 1e-4,
        "num_steps": 128,
        "batch_size": 32,
        "selected_groups": SELECTED_GROUPS,
    }
    print("POST /v1/finetune/jobs")
    print(json.dumps(payload, indent=2))

    resp = requests.post(f"{BASE_URL}/v1/finetune/jobs", json=payload)
    print(f"Status: {resp.status_code}")
    resp.raise_for_status()
    result = resp.json()
    print(json.dumps(result, indent=2))
    job_id = result["job_id"]

    # 3. Poll job status
    print("\n[3] Poll Job Status")
    job_detail = _poll_job(job_id)
    print(json.dumps(job_detail, indent=2, default=str))

    # 4. Publish + 5. Infer (if completed)
    if job_detail.get("status") == "completed":
        model_path = _publish_model(job_id)
        _infer_model(model_path)
    else:
        raise RuntimeError(f"训练任务未完成，最终状态: {job_detail.get('status')}")

    # 6. Fetch logs
    print("\n[6] Fetch Job Logs (tail=30)")
    resp = requests.get(f"{BASE_URL}/v1/finetune/jobs/{job_id}/logs", params={"tail": 30})
    print(f"GET /v1/finetune/jobs/{job_id}/logs -> {resp.status_code}")
    print(resp.text)

    _print_section("DEMO COMPLETED")


if __name__ == "__main__":
    demo()
