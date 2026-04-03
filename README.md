# Chronos-2 模型微调服务

基于 FastAPI 的服务，用于使用 LoRA 或全量微调模式微调亚马逊 Chronos-2 时间序列预测模型。

## 项目概览

本项目提供一个 REST API 来提交 Chronos-2 模型的微调任务。当前进度（第 5 步）已接入查询接口、取消接口与真实的 Chronos-2 微调，支持：

- **任务创建**：提交经过参数验证的微调任务
- **任务持久化**：在 SQLite 数据库中存储任务元数据
- **任务目录管理**：自动创建输出目录并保存请求清单
- **后台异步训练**：后台 worker 自动消费任务队列
- **真实 Chronos-2 微调**：使用官方 Chronos-2 库进行实际的模型微调
- **进度跟踪与 Callback**：通过自定义 callback 在训练过程中实时更新进度
- **健康检查**：简单的健康检查端点用于服务监控

### 当前功能

**已实现**：

- ✅ 后台异步任务队列（本地内存队列）
- ✅ 后台 worker 线程（串行处理任务）
- ✅ **真实 Chronos-2 微调**（官方 `fit()` 接口）
- ✅ **自定义 callback 机制**（训练过程中更新数据库和日志）
- ✅ 数据集加载（支持 CSV 和 Parquet 格式）
- ✅ 任务状态流转（queued → running → completed/failed）
- ✅ 进度跟踪（current_step, max_steps, last_loss）
- ✅ 任务查询接口（详情 / 结果 / 日志）
- ✅ 任务取消接口（协作式取消）
- ✅ CPU/CUDA 自动设备检测

**暂未实现/计划中**：

- 分布式训练（多GPU）
- 分布式 worker（当前为单线程）
- 高级数据预处理和特征工程
- 模型评估和验证指标

### 技术架构

```text
HTTP 请求
    ↓
FastAPI 路由 (POST /v1/finetune/jobs)
    ↓
参数验证 → 数据库入库 → 加入队列
    ↓
后台 Worker 线程
    ↓
轮询队列 → 消费任务 → 加载数据 → Chronos-2 fit()
    ↓
Callback 更新进度 → 保存模型 → 更新状态
```

- **任务队列**：基于 Python 标准库 `queue.Queue`（易后续扩展为 RabbitMQ/Redis）
- **数据库**：SQLite + SQLAlchemy ORM
- **API 框架**：FastAPI + Uvicorn
- **模型库**：chronos-forecasting 2.2.2（LoRA 需安装 peft）
- **数据处理**：Pandas + PyArrow

## 需求

- Python 3.11+
- pip (或 uv)
- PyTorch（自动作为 chronos-forecasting 依赖安装）

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

> **注意**：首次安装时，chronos-forecasting 和 PyTorch 的下载和安装可能需要几分钟到十几分钟，
> 取决于网络速度和机器配置。

## 使用方法

### 启动服务

```bash
# 方式 A：直接使用 uvicorn（端口不会读取 .env，需手动指定）
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 方式 B：用 python -m uvicorn 启动（同样需要手动指定端口）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 方式 C：使用项目内的启动入口（会读取 .env 中的 HOST/PORT）
python -m app.main
```

**部署到服务器并允许远程访问时：**

- 将 `HOST` 设为 `0.0.0.0`（监听所有网卡）。
- 仅本机访问则保持 `127.0.0.1`。
- 对外访问时请配合防火墙/反向代理限制来源。

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

最小请求示例（仅需 train_data_path 和 prediction_length）：

```bash
curl -X POST http://127.0.0.1:8000/v1/finetune/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "train_data_path": "/path/to/train.csv",
    "prediction_length": 96
  }'
```

完整请求示例（包含所有可选参数）：

```bash
curl -X POST http://127.0.0.1:8000/v1/finetune/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "train_data_path": "/path/to/train.csv",
    "val_data_path": "/path/to/val.csv",
    "prediction_length": 96,
    "context_length": 512,
    "finetune_mode": "lora",
    "learning_rate": 0.0001,
    "num_steps": 1000,
    "batch_size": 32,
    "logging_steps": 100,
    "output_root": "/custom/path",
    "finetuned_ckpt_name": "finetuned-ckpt",
    "device": "cpu",
    "selected_columns": ["target"]
  }'
```

响应（立即返回，同步返回 job_id 和状态）：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `train_data_path` | str | **必需** | 训练数据文件路径（CSV 或 Parquet） |
| `val_data_path` | str \| null | null | 验证数据文件路径（可选） |
| `prediction_length` | int | **必需** | 预测长度（时间步） |
| `context_length` | int | 512 | 上下文长度（输入时间步） |
| `finetune_mode` | str | "lora" | 微调模式：`"lora"` 或 `"full"` |
| `learning_rate` | float | 1e-4 | 学习率 |
| `num_steps` | int | 1000 | 训练总步数 |
| `batch_size` | int | 32 | 批处理大小 |
| `logging_steps` | int | 100 | 日志间隔（步） |
| `output_root` | str \| null | null | 输出根目录（default: `./artifacts`） |
| `finetuned_ckpt_name` | str | "finetuned-ckpt" | 微调模型保存名称 |
| `device` | str | "cpu" | 设备：`"cpu"` 或 `"cuda"`（自动检测） |
| `selected_columns` | list[str] \| null | null | 仅使用指定列（为空则使用全部列） |

**任务状态说明**：

- `queued`: 任务已创建，等待 worker 消费
- `running`: 后台 worker 正在执行 Chronos-2 微调
- `completed`: 训练成功完成
- `failed`: 训练过程中出现错误
- `cancelled`: 任务已取消（协作式取消）

### 支持的数据格式

当前版本的数据读取逻辑是：读取表格 →（可选）按 `selected_columns` 选择列 → 转为 `float32` 数组。
因此数据格式需要满足下面要求：

- 支持 CSV / Parquet
- 每一列被视为一个变量（variates），每一行是一个时间步
- 所选列必须是数值列（否则转换为 `float32` 会失败）
- 若未传 `selected_columns`，将使用全部列
- 当前版本不会解析 `item_id` / `timestamp` 等长表字段；若有这些列，请在上传前转换为“宽表”或用 `selected_columns` 排除

#### CSV 示例（宽表）

```csv
value1,value2,value3
100.5,200.1,300.0
101.3,202.5,299.6
99.8,201.2,301.4
```

#### Parquet 格式

与 CSV 相同的列结构（宽表、数值列）。Pandas 会自动解析 Parquet 文件。

### 查询任务状态

当前版本已支持任务查询接口，包括详情、结果和日志：

1) 查询任务详情

```bash
curl http://127.0.0.1:8000/v1/finetune/jobs/<job_id>
```

响应示例：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:10Z",
  "finished_at": null,
  "progress": {
    "current_step": 120,
    "max_steps": 1000,
    "last_loss": 0.5321
  },
  "error_message": null,
  "log_path": "./artifacts/550e8400-e29b-41d4-a716-446655440000/train.log",
  "model_path": null
}
```

1) 查询任务结果（仅完成后可用）

```bash
curl http://127.0.0.1:8000/v1/finetune/jobs/<job_id>/result
```

响应示例：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "output_dir": "./artifacts/550e8400-e29b-41d4-a716-446655440000",
  "model_path": "./artifacts/550e8400-e29b-41d4-a716-446655440000/finetuned-ckpt",
  "metrics": {}
}
```

1) 查询任务日志（支持 tail）

```bash
# 返回完整日志
curl http://127.0.0.1:8000/v1/finetune/jobs/<job_id>/logs

# 返回最后 200 行
curl "http://127.0.0.1:8000/v1/finetune/jobs/<job_id>/logs?tail=200"
```

1) 任务列表（可选）

```bash
curl "http://127.0.0.1:8000/v1/finetune/jobs?limit=20"
```

### 取消任务

当前版本支持**协作式取消**（cooperative cancellation），不会强制杀进程：

- `queued`：直接标记为 `cancelled`，任务不会被执行。
- `running`：设置 `cancel_requested=true`，训练回调/训练循环检测到后尽快中止，并最终标记为 `cancelled`。

**取消接口：**

```bash
curl -X POST http://127.0.0.1:8000/v1/finetune/jobs/<job_id>/cancel
```

响应示例：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "cancel_requested": true,
  "message": "已请求取消，等待训练停止"
}
```

result 接口与 detail 接口的区别

- `detail`（`/v1/finetune/jobs/{job_id}`）：用于查询任务实时状态、进度和错误信息，任何状态都可用。
- `result`（`/v1/finetune/jobs/{job_id}/result`）：仅在任务完成后返回模型输出与指标，否则返回 4xx。

### 任务目录结构

任务创建后，会在 `artifacts/` 下生成以下结构：

```txt
artifacts/
└── <job_id>/
    ├── request.json          # 保存的请求参数
    ├── train.log             # 训练日志（实时写入）
    └── finetuned-ckpt/       # 微调后的模型（训练完成后）
        ├── model.pt
        ├── config.json
        └── ...
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

```

## 目录结构

```text
ts_model_train_and_finetune/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用工厂 + 生命周期管理
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   ├── health.py        # 健康检查端点
│   │   └── finetune.py      # 任务创建端点
│   ├── callbacks/           # 回调机制（新）
│   │   ├── __init__.py
│   │   └── progress_callback.py  # 训练进度回调
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
│   ├── services/            # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── queue_service.py # 任务队列管理
│   │   ├── job_service.py   # 任务业务逻辑
│   │   ├── trainer_service.py # 真实 Chronos-2 微调
│   │   └── dataset_service.py # 数据集加载与转换
│   └── workers/             # 后台 worker
│       ├── __init__.py
│       └── trainer_worker.py # 训练 worker 线程
│
├── tests/
│   ├── __init__.py
│   ├── test_create_job.py    # 第 1 步测试（任务创建）
│   ├── test_worker_flow.py   # 第 2 步测试（异步工作流）
│   └── test_trainer_service.py # 第 3 步测试（真实训练）
│
├── pyproject.toml
├── README.md
└── .gitignore
```

## 数据库架构

`finetune_jobs` 表存储任务元数据，字段如下：

| 字段 | 类型 | 可空 | 默认值 |
| ---- | ---- | ---- | ------ |
| id | VARCHAR(36) | 否 | - |
| status | VARCHAR(20) | 否 | "queued" |
| request_json | TEXT | 否 | - |
| created_at | DATETIME | 否 | 当前时间 |
| started_at | DATETIME | 是 | NULL |
| finished_at | DATETIME | 是 | NULL |
| output_dir | VARCHAR(512) | 否 | - |
| log_path | VARCHAR(512) | 否 | - |
| model_path | VARCHAR(512) | 是 | NULL |
| error_message | TEXT | 是 | NULL |
| current_step | INTEGER | 否 | 0 |
| max_steps | INTEGER | 否 | 0 |
| last_loss | FLOAT | 是 | NULL |
| cancel_requested | BOOLEAN | 否 | False |

## 前端 API 文档

### 基础信息

Base URL: `http://127.0.0.1:8000`  
Content-Type: `application/json`  
时间字段: ISO 8601（UTC）  
认证: 无

### 任务状态

`queued` / `running` / `completed` / `failed` / `cancelled`

### GET /health

用途: 健康检查  
响应 200:

```json
{
  "status": "ok"
}
```

### POST /v1/finetune/jobs

用途: 创建训练任务  
请求体:

| 字段 | 类型 | 默认值 | 必需 | 说明 |
| ---- | ---- | ------ | ---- | ---- |
| train_data_path | str | - | **是** | 训练数据路径 |
| val_data_path | str \| null | null | 否 | 验证数据路径 |
| prediction_length | int | - | **是** | 预测长度（正整数） |
| context_length | int | 512 | 否 | 上下文长度（正整数） |
| finetune_mode | str | lora | 否 | `"lora"` 或 `"full"` |
| learning_rate | float | 0.0001 | 否 | 学习率（正数） |
| num_steps | int | 1000 | 否 | 训练总步数（正整数） |
| batch_size | int | 32 | 否 | 批处理大小（正整数） |
| logging_steps | int | 100 | 否 | 日志间隔（正整数） |
| output_root | str \| null | null | 否 | 输出根目录（为空则用 `./artifacts`） |
| finetuned_ckpt_name | str | finetuned-ckpt | 否 | 模型保存目录名 |
| device | str | cpu | 否 | `"cpu"` 或 `"cuda"` |
| selected_columns | list[str] \| null | null | 否 | 使用指定列；不传则使用全部列 |

响应 201:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

常见错误: 422（参数校验失败）

### GET /v1/finetune/jobs

用途: 查询最近任务列表  
查询参数: `limit` (int, 默认 20)  
响应 200:

```json
{
  "items": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "running",
      "created_at": "2024-01-01T12:00:00Z",
      "started_at": "2024-01-01T12:00:10Z",
      "finished_at": null
    }
  ]
}
```

### GET /v1/finetune/jobs/{job_id}

用途: 查询任务详情与进度  
响应 200:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:10Z",
  "finished_at": null,
  "progress": {
    "current_step": 120,
    "max_steps": 1000,
    "last_loss": 0.5321
  },
  "error_message": null,
  "log_path": "./artifacts/550e8400-e29b-41d4-a716-446655440000/train.log",
  "model_path": null
}
```

常见错误: 404（任务不存在）

### GET /v1/finetune/jobs/{job_id}/result

用途: 查询任务结果（仅完成后可用）  
响应 200:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "output_dir": "./artifacts/550e8400-e29b-41d4-a716-446655440000",
  "model_path": "./artifacts/550e8400-e29b-41d4-a716-446655440000/finetuned-ckpt",
  "metrics": {}
}
```

常见错误: 409（任务未完成），404（任务不存在）

### GET /v1/finetune/jobs/{job_id}/logs

用途: 查询日志  
查询参数: `tail` (int, 可选，仅返回最后 N 行)  
响应 200（`text/plain`）:

```text
训练开始: 任务 550e8400-e29b-41d4-a716-446655440000
[步骤 1], 损失=0.98
...
```

常见错误: 404（任务或日志文件不存在）

### POST /v1/finetune/jobs/{job_id}/cancel

用途: 请求取消任务（协作式取消）  
响应 200:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "cancel_requested": true,
  "message": "已请求取消，等待训练停止"
}
```

常见错误: 409（状态不允许取消），404（任务不存在）

## 测试

运行完整测试套件：

```bash
pytest tests/ -v
```

运行指定测试模块：

```bash
# 第 1 步：任务创建接口
pytest tests/test_create_job.py -v

# 第 2 步：异步工作流
pytest tests/test_worker_flow.py -v

# 第 3 步：真实训练（含 mock）
pytest tests/test_trainer_service.py -v

# 第 4 步：任务查询接口
pytest tests/test_query_api.py -v

# 第 5 步：任务取消接口
pytest tests/test_cancel_job.py -v
```

带覆盖率的测试：

```bash
pytest tests/ --cov=app
```

运行特定测试：

```bash
pytest tests/test_create_job.py::test_create_finetune_job_success -v
```

## 项目设计说明

- **简洁设计**：最小化抽象，直接的实现
- **类型注解**：所有 Python 代码都包含类型注解，便于 IDE 支持和代码维护
- **路径处理**：全部使用 `pathlib.Path` 确保跨平台兼容性
- **真实训练**：使用官方 Chronos-2 `fit()` 接口进行实际的模型微调
- **Callback 机制**：自定义 callback 在训练过程中实时更新数据库和日志

## 项目进度

**第 1 步（已完成）**：创建任务接口，仅入库，不启动训练

- ✅ 参数校验
- ✅ 数据库入库
- ✅ 产物目录管理
- ✅ 返回 job_id

**第 2 步（已完成）**：异步训练过程（使用假训练器）

- ✅ 本地内存队列
- ✅ 后台 worker 线程
- ✅ 任务自动入队和消费
- ✅ 状态流转（queued → running → completed/failed）
- ✅ 进度跟踪

**第 3 步（已完成）**：接入真实 Chronos-2 微调并实现 callback

- ✅ 真实模型加载和微调
- ✅ 数据集格式支持（CSV/Parquet）
- ✅ 自定义 callback 机制
- ✅ 进度实时更新
- ✅ 模型保存

**第 4 步（已完成）**：任务查询接口

- ✅ 任务查询端点（详情 / 结果 / 日志）

**第 5 步（已完成）**：取消与辅助接口

- ✅ 任务取消接口（协作式取消）

## 许可证

内部项目，仅供模型微调研究使用。
