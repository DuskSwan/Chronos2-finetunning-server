# Chronos-2 模型微调服务

基于 FastAPI 的服务，用于使用 LoRA 或全量微调模式微调亚马逊 Chronos-2 时间序列预测模型。

## 项目概览

本项目提供一个 REST API 来提交 Chronos-2 模型的微调任务。当前进度（第 2 步）支持：

- **任务创建**：提交经过参数验证的微调任务
- **任务持久化**：在 SQLite 数据库中存储任务元数据
- **任务目录管理**：自动创建输出目录并保存请求清单
- **后台异步训练**：后台 worker 自动消费任务队列
- **任务进度跟踪**：实时更新训练步数、损失等信息
- **健康检查**：简单的健康检查端点用于服务监控

### 当前功能

**已实现**：

- ✅ 后台异步任务队列（本地内存队列）
- ✅ 后台 worker 线程（串行处理任务）
- ✅ 假训练器（5步模拟训练，每步0.2~0.5秒）
- ✅ 任务状态流转（queued → running → completed/failed）
- ✅ 进度跟踪（current_step, max_steps, last_loss）

**暂未实现/计划中**：

- 真实 Chronos-2 模型训练调用
- 任务取消功能
- 回调（callback）机制
- 查询接口大幅扩充
- 分布式 worker（当前为单线程）

### 技术架构

```
HTTP 请求
    ↓
FastAPI 路由 (POST /v1/finetune/jobs)
    ↓
参数验证 → 数据库入库 → 加入队列
    ↓
后台 Worker 线程
    ↓
轮询队列 → 消费任务 → 假训练 → 更新状态
```

- **任务队列**：基于 Python 标准库 `queue.Queue`（易后续扩展为 RabbitMQ/Redis）
- **数据库**：SQLite + SQLAlchemy ORM
- **API 框架**：FastAPI + Uvicorn

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

服务启动时会自动：
1. 初始化 SQLite 数据库
2. 创建必要目录（artifacts, logs）
3. 启动后台 worker 线程

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

响应（立即返回，不等待训练完成）：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

**任务状态说明**：

- `queued`: 任务已创建，等待 worker 消费
- `running`: 后台 worker 正在训练
- `completed`: 训练成功完成
- `failed`: 训练过程中出现错误
- `cancelled`: 任务已取消（暂未实现）

### 查询任务状态（调试用）

当前版本暂不支持直接的查询接口，但可以通过检查数据库或日志目录来了解任务状态：

```bash
# 检查数据库中的任务信息
sqlite3 ./data/finetune.db "SELECT id, status, started_at, finished_at FROM finetune_jobs;"

# 查看训练日志
tail -f ./artifacts/<job_id>/train.log
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
│   ├── main.py              # FastAPI 应用工厂 + 生命周期管理
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
│   ├── schemas/             # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── request.py       # 请求 schema
│   │   └── response.py      # 响应 schema
│   ├── services/            # 业务逻辑层（新）
│   │   ├── __init__.py
│   │   ├── queue_service.py # 任务队列管理
│   │   ├── job_service.py   # 任务业务逻辑
│   │   └── trainer_service.py # 假训练器
│   └── workers/             # 后台 worker（新）
│       ├── __init__.py
│       └── trainer_worker.py # 训练 worker 线程
│
├── tests/
│   ├── __init__.py
│   ├── test_create_job.py    # 第 1 步测试
│   └── test_worker_flow.py   # 第 2 步测试（新）
│
├── pyproject.toml
├── README.md
└── .gitignore
```
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
