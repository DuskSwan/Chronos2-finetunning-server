# Chronos-2 模型微调服务

基于 FastAPI 的服务，用于使用 LoRA 或全量微调模式微调亚马逊 Chronos-2 时间序列预测模型。

## 项目概览

本项目提供一个 REST API 来提交 Chronos-2 模型的微调任务。在第一阶段（步骤 1），服务支持：

- **任务创建**：提交经过参数验证的微调任务
- **任务持久化**：在 SQLite 数据库中存储任务元数据
- **任务目录管理**：自动创建输出目录并保存请求清单
- **健康检查**：简单的健康检查端点用于服务监控

### 当前功能

本步骤仅实现任务提交和持久化层。**不包含**：

- 后台训练 worker 进程
- 异步训练执行
- 真实的 Chronos-2 模型训练调用
- 任务取消功能
- 回调机制
- 任务查询端点（除了健康检查）

## 需求

- Python 3.11+
- pip (或 uv)

## 安装

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 上使用: .venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -e .
# 或带测试依赖：
pip install -e ".[dev]"
```

## 使用方法

### 启动服务

```bash
# 直接使用 uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 或使用 python -m
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

服务将在 `http://127.0.0.1:8000` 启动。

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

响应：

```json
{
  "status": "ok"
}
```

### 创建微调任务

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

响应：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

## 配置

配置通过环境变量或 `.env` 文件管理：

```env
# 服务器设置
HOST=127.0.0.1
PORT=8000

# 数据库
SQLITE_DB_PATH=./data/finetune.db

# 路径
ARTIFACTS_ROOT=./artifacts
LOGS_ROOT=./logs

# 模型
DEFAULT_MODEL_ID=amazon/chronos-2
```

## 目录结构

```bash
ts_model_train_and_finetune/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用工厂
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   ├── health.py        # 健康检查端点
│   │   └── finetune.py      # 任务创建端点
│   ├── core/                # 核心配置
│   │   ├── __init__.py
│   │   ├── config.py        # 设置管理
│   │   ├── paths.py         # 路径工具
│   │   └── enums.py         # 状态枚举
│   ├── db/                  # 数据库层
│   │   ├── __init__.py
│   │   ├── session.py       # SQLAlchemy 会话设置
│   │   ├── models.py        # ORM 模型
│   │   ├── crud.py          # CRUD 操作
│   │   └── init_db.py       # 数据库初始化
│   └── schemas/             # Pydantic schemas
│       ├── __init__.py
│       ├── request.py       # 请求 schema
│       └── response.py      # 响应 schema
├── tests/
│   └── test_create_job.py   # 任务创建测试
├── pyproject.toml
├── README.md
└── .gitignore
```

## Database Schema

The `finetune_jobs` table stores job metadata with the following fields:

| Field | Type | Nullable | Default |
| ----- | ---- | -------- | ------- |
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
| ----- | ---- | ------- | -------- | ----- |
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
