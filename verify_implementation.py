#!/usr/bin/env python
"""Verification script for the fine-tuning job implementation."""

import sqlite3
from pathlib import Path
import json

def verify_database():
    """Verify database and records."""
    db_path = Path("data/finetune.db")
    
    if not db_path.exists():
        print("❌ Database file does not exist")
        return False
    
    print("✓ Database file exists")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='finetune_jobs'"
    )
    if not cursor.fetchone():
        print("❌ Table 'finetune_jobs' does not exist")
        conn.close()
        return False
    
    print("✓ Table 'finetune_jobs' exists")
    
    # Get record count
    cursor.execute("SELECT COUNT(*) FROM finetune_jobs")
    count = cursor.fetchone()[0]
    print(f"✓ Database contains {count} job record(s)")
    
    # Get the last few records
    print("\n=== Last 3 Database Records ===")
    cursor.execute(
        "SELECT id, status, created_at, output_dir FROM finetune_jobs ORDER BY created_at DESC LIMIT 3"
    )
    for row in cursor.fetchall():
        print(f"  Job ID: {row[0]}")
        print(f"  Status: {row[1]}")
        print(f"  Created: {row[2]}")
        print(f"  Output Dir: {row[3]}")
        print()
    
    conn.close()
    return True


def verify_artifacts():
    """Verify artifact directories and request.json files."""
    artifacts_root = Path("artifacts")
    
    if not artifacts_root.exists():
        print("❌ artifacts/ directory does not exist")
        return False
    
    print("✓ artifacts/ directory exists")
    
    # Count job directories
    job_dirs = list(artifacts_root.glob("*"))
    job_dirs = [d for d in job_dirs if d.is_dir()]
    
    if not job_dirs:
        print("⚠ No job directories found in artifacts/")
        return True
    
    print(f"✓ Found {len(job_dirs)} job directories")
    
    # Check request.json files
    for job_dir in job_dirs[:3]:
        request_json = job_dir / "request.json"
        if request_json.exists():
            with open(request_json) as f:
                data = json.load(f)
                print(f"  ✓ {job_dir.name}/request.json exists")
                print(f"    - train_data_path: {data.get('train_data_path')}")
                print(f"    - prediction_length: {data.get('prediction_length')}")
        else:
            print(f"  ❌ {job_dir.name}/request.json not found")
    
    return True


def verify_logs():
    """Verify logs directory exists."""
    logs_root = Path("logs")
    
    if not logs_root.exists():
        print("❌ logs/ directory does not exist")
        return False
    
    print("✓ logs/ directory exists")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("VERIFICATION REPORT")
    print("=" * 50)
    
    print("\n[1] Database Verification")
    print("-" * 50)
    db_ok = verify_database()
    
    print("\n[2] Artifacts Verification")
    print("-" * 50)
    artifacts_ok = verify_artifacts()
    
    print("\n[3] Logs Verification")
    print("-" * 50)
    logs_ok = verify_logs()
    
    print("\n" + "=" * 50)
    if db_ok and artifacts_ok and logs_ok:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("⚠ Some verifications failed")
    print("=" * 50)
