#!/usr/bin/env python
"""End-to-end demo using mock_train_data.csv."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
DATA_FILE = Path("mock_train_data.csv").resolve()
SELECTED_COLUMNS = ["value1", "value2", "value3"]


def _print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _require_data_file() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {DATA_FILE}. 请确认 mock_train_data.csv 在仓库根目录。"
        )


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


def demo() -> None:
    _require_data_file()

    _print_section("CHRONOS-2 FINE-TUNING JOB SERVICE - DEMO")

    # 1. Health Check
    print("[1] Health Check")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"GET /health -> {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    # 2. Create Job (mock_train_data.csv + selected_columns)
    print("\n[2] Create Fine-tuning Job")
    payload = {
        "train_data_path": str(DATA_FILE),
        "prediction_length": 1,
        "context_length": 2,
        "finetune_mode": "lora",
        "learning_rate": 1e-4,
        "num_steps": 1,
        "batch_size": 1,
        "logging_steps": 1,
        "finetuned_ckpt_name": "finetuned-ckpt",
        "device": "cpu",
        "selected_columns": SELECTED_COLUMNS,
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

    # 4. Get result (if completed)
    if job_detail.get("status") == "completed":
        print("\n[4] Fetch Job Result")
        resp = requests.get(f"{BASE_URL}/v1/finetune/jobs/{job_id}/result")
        print(f"GET /v1/finetune/jobs/{job_id}/result -> {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))

    # 5. Fetch logs
    print("\n[5] Fetch Job Logs (tail=30)")
    resp = requests.get(f"{BASE_URL}/v1/finetune/jobs/{job_id}/logs", params={"tail": 30})
    print(f"GET /v1/finetune/jobs/{job_id}/logs -> {resp.status_code}")
    print(resp.text)

    _print_section("✓ DEMO COMPLETED")


if __name__ == "__main__":
    demo()
