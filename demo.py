#!/usr/bin/env python
"""Complete end-to-end demo of the fine-tuning job API."""

import requests
import json
from pathlib import Path
import time

BASE_URL = "http://127.0.0.1:8000"

def demo():
    print("=" * 70)
    print("CHRONOS-2 FINE-TUNING JOB SERVICE - END-TO-END DEMO")
    print("=" * 70)
    
    # 1. Health Check
    print("\n[1] Testing Health Check Endpoint")
    print("-" * 70)
    resp = requests.get(f"{BASE_URL}/health")
    print(f"GET /health")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")
    
    # 2. Create Job (Minimal)
    print("\n[2] Creating Fine-tuning Job (Minimal Parameters)")
    print("-" * 70)
    payload = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96
    }
    print(f"POST /v1/finetune/jobs")
    print(f"Request Body: {json.dumps(payload, indent=2)}")
    
    resp = requests.post(f"{BASE_URL}/v1/finetune/jobs", json=payload)
    print(f"\nStatus: {resp.status_code}")
    result = resp.json()
    job_id = result['job_id']
    print(f"Response: {json.dumps(result, indent=2)}")
    
    # 3. Verify Artifact Directory
    print("\n[3] Verifying Artifact Directory Creation")
    print("-" * 70)
    task_dir = Path("artifacts") / job_id
    print(f"Task Directory: {task_dir}")
    print(f"Exists: {task_dir.exists()}")
    
    request_json_file = task_dir / "request.json"
    if request_json_file.exists():
        print(f"✓ request.json exists")
        with open(request_json_file) as f:
            saved_request = json.load(f)
            print(f"Content (with applied defaults):")
            print(json.dumps(saved_request, indent=2))
    
    # 4. Create Job (Full Parameters)
    print("\n[4] Creating Fine-tuning Job (Full Parameters)")
    print("-" * 70)
    payload = {
        "model_id": "amazon/chronos-2",
        "train_data_path": "/path/to/train.csv",
        "val_data_path": "/path/to/val.csv",
        "prediction_length": 96,
        "context_length": 512,
        "finetune_mode": "lora",
        "learning_rate": 0.0001,
        "num_steps": 1000,
        "batch_size": 32,
        "logging_steps": 100,
        "output_root": None,
        "finetuned_ckpt_name": "finetuned-ckpt",
        "device": "cpu"
    }
    print(f"POST /v1/finetune/jobs")
    print(f"Request Body: {json.dumps(payload, indent=2)}")
    
    resp = requests.post(f"{BASE_URL}/v1/finetune/jobs", json=payload)
    print(f"\nStatus: {resp.status_code}")
    result = resp.json()
    job_id_2 = result['job_id']
    print(f"Response: {json.dumps(result, indent=2)}")
    
    # 5. Verify Database
    print("\n[5] Verifying Database Records")
    print("-" * 70)
    import sqlite3
    conn = sqlite3.connect("data/finetune.db")
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, status, created_at FROM finetune_jobs ORDER BY created_at DESC LIMIT 2"
    )
    rows = cursor.fetchall()
    for row in rows:
        print(f"  Job ID: {row[0]}")
        print(f"  Status: {row[1]}")
        print(f"  Created: {row[2]}")
        print()
    
    conn.close()
    
    # 6. Test Validation (Invalid Finetune Mode)
    print("\n[6] Testing Input Validation (Invalid finetune_mode)")
    print("-" * 70)
    payload = {
        "train_data_path": "/path/to/train.csv",
        "prediction_length": 96,
        "finetune_mode": "invalid"
    }
    print(f"POST /v1/finetune/jobs (with invalid finetune_mode)")
    resp = requests.post(f"{BASE_URL}/v1/finetune/jobs", json=payload)
    print(f"Status: {resp.status_code} (Expected: 422)")
    print(f"Error details: {json.dumps(resp.json(), indent=2)}")
    
    print("\n" + "=" * 70)
    print("✓ END-TO-END DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    demo()
