# Chronos-2 Model Fine-tuning Service

A FastAPI-based service for fine-tuning Amazon Chronos-2 time series forecasting models using LoRA or full fine-tuning modes.

## Project Overview

This project provides a REST API to submit fine-tuning jobs for Chronos-2 models. In this first phase (Step 1), the service supports:

- **Job Creation**: Submit fine-tuning jobs with validated parameters
- **Job Persistence**: Store job metadata in SQLite database
- **Task Directory Management**: Automatically create output directories and save request manifests
- **Health Checking**: Simple health check endpoint for service monitoring

### Current Capabilities

This step implements the task submission and persistence layer only. It does **NOT** include:

- Background training worker processes
- Asynchronous training execution
- Real Chronos-2 model training invocations
- Job cancellation
- Callback mechanisms
- Job query endpoints (except health check)

## Requirements

- Python 3.11+
- pip (or uv)

## Installation

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -e .
# Or with dev dependencies for testing:
pip install -e ".[dev]"
```

## Usage

### Starting the Service

```bash
# Using uvicorn directly
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Or using python -m
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The service will start at `http://127.0.0.1:8000`.

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "ok"
}
```

### Create a Fine-tuning Job

```bash
curl -X POST http://127.0.0.1:8000/v1/finetune/jobs \
  -H "Content-Type: application/json" \
  -d '{
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
    "finetuned_ckpt_name": "finetuned-ckpt",
    "device": "cpu"
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

## Configuration

Configuration is managed via environment variables or a `.env` file:

```env
# Server settings
HOST=127.0.0.1
PORT=8000

# Database
SQLITE_DB_PATH=./data/finetune.db

# Paths
ARTIFACTS_ROOT=./artifacts
LOGS_ROOT=./logs

# Model
DEFAULT_MODEL_ID=amazon/chronos-2
```

## Directory Structure

```
ts_model_train_and_finetune/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application factory
│   ├── api/                 # API routes
│   │   ├── __init__.py
│   │   ├── health.py        # Health check endpoint
│   │   └── finetune.py      # Job creation endpoint
│   ├── core/                # Core configurations
│   │   ├── __init__.py
│   │   ├── config.py        # Settings management
│   │   ├── paths.py         # Path utilities
│   │   └── enums.py         # Status enumerations
│   ├── db/                  # Database layer
│   │   ├── __init__.py
│   │   ├── session.py       # SQLAlchemy session setup
│   │   ├── models.py        # ORM models
│   │   ├── crud.py          # CRUD operations
│   │   └── init_db.py       # Database initialization
│   └── schemas/             # Pydantic schemas
│       ├── __init__.py
│       ├── request.py       # Request schemas
│       └── response.py      # Response schemas
├── tests/
│   └── test_create_job.py   # Job creation tests
├── pyproject.toml
├── README.md
└── .gitignore
```

## Database Schema

The `finetune_jobs` table stores job metadata with the following fields:

| Field | Type | Nullable | Default |
|-------|------|----------|---------|
| id | VARCHAR(36) | No | - |
| status | VARCHAR(20) | No | "queued" |
| request_json | TEXT | No | - |
| created_at | DATETIME | No | Now |
| started_at | DATETIME | Yes | NULL |
| finished_at | DATETIME | Yes | NULL |
| output_dir | VARCHAR(512) | No | - |
| log_path | VARCHAR(512) | No | - |
| model_path | VARCHAR(512) | Yes | NULL |
| error_message | TEXT | Yes | NULL |
| current_step | INTEGER | No | 0 |
| max_steps | INTEGER | No | 0 |
| last_loss | FLOAT | Yes | NULL |
| cancel_requested | BOOLEAN | No | False |

## API Endpoints

### GET /health

Health check endpoint.

**Response 200:**
```json
{
  "status": "ok"
}
```

### POST /v1/finetune/jobs

Create a new fine-tuning job.

**Request Body:**

| Field | Type | Default | Required | Notes |
|-------|------|---------|----------|-------|
| model_id | string | amazon/chronos-2 | No | - |
| train_data_path | string | - | Yes | Path to training data |
| val_data_path | string | null | No | Path to validation data |
| prediction_length | integer | - | Yes | Must be positive |
| context_length | integer | 512 | No | Must be positive |
| finetune_mode | string | lora | No | "lora" or "full" |
| learning_rate | float | 0.0001 | No | Must be positive |
| num_steps | integer | 1000 | No | Must be positive |
| batch_size | integer | 32 | No | Must be positive |
| logging_steps | integer | 100 | No | Must be positive |
| output_root | string | null | No | Uses artifacts_root if null |
| finetuned_ckpt_name | string | finetuned-ckpt | No | - |
| device | string | cpu | No | "cpu" or "cuda" |

**Response 201:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

**Request Validation:**
- `train_data_path`: Required, non-empty string
- `prediction_length`: Required, positive integer
- `context_length`: Positive integer
- `num_steps`: Positive integer
- `batch_size`: Positive integer
- `logging_steps`: Positive integer
- `learning_rate`: Positive float
- `finetune_mode`: One of "lora" or "full"

## Testing

Run the test suite:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=app
```

Run specific test:

```bash
pytest tests/test_create_job.py::test_create_finetune_job_success -v
```

## Project Structure Notes

- **Simple Design**: Minimal abstractions, straightforward implementations
- **Type Hints**: All Python code includes type annotations for better IDE support and maintainability
- **Path Handling**: Uses `pathlib.Path` exclusively for cross-platform compatibility
- **No Training Logic**: This step focuses solely on API and persistence layers

## Next Steps (Future Phases)

- Phase 2: Implement background worker and async training execution
- Phase 3: Add job query and cancellation endpoints
- Phase 4: Implement real Chronos-2 model training
- Phase 5: Add callback mechanisms and monitoring

## License

Internal project for model fine-tuning research.
