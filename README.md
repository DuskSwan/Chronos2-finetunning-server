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
- **工具接口**：提供数据分析工具，如相关性矩阵计算

### 当前功能

**已实现**：

- ✅ 后台异步任务队列（本地内存队列）
- ✅ 后台 worker 线程（串行处理任务）
- ✅ **真实 Chronos-2 微调**（官方 `fit()` 接口）
- ✅ **自定义 callback 机制**（训练过程中更新数据库和日志）
- ✅ 数据集加载（支持 CSV 和 Parquet 格式）
- ✅ 任务状态流转（queued → running → completed/failed）
- ✅ 进度跟踪（current_step, max_steps, last_loss）
- ✅ 按预测目标返回 loss 曲线（`metrics.<target> = [loss...]`）
- ✅ 任务查询接口（详情 / 日志）
- ✅ 任务取消接口（协作式取消）
- ✅ 模型发布接口（按 `user_id + job_id + version` 发布，可重复调用覆盖）
- ✅ 规范兼容接口（`/api/v1/train_jobs*`，统一 `code/message/data`）
- ✅ CPU/CUDA 自动设备检测
- ✅ **工具接口**（相关性矩阵计算）

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
uvicorn app.main:app --host 127.0.0.1 --port 8011

# 方式 B：用 python -m uvicorn 启动（同样需要手动指定端口）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8011 --reload

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
3. 恢复数据库中 `queued`/`running` 任务并重新入队（`running` 会从头重新训练）
4. 启动后台 worker 线程

### 健康检查

```bash
curl http://127.0.0.1:8011/health
```

响应：

```json
{
  "status": "ok"
}
```

### 工具接口

#### 计算相关性矩阵

计算指定列之间的相关性矩阵，支持多种相关性计算方法。

```bash
curl -X POST http://127.0.0.1:8011/v1/tools/correlation \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/path/to/data.csv",
    "columns": ["a", "c"],
    "method": "pearson"
  }'
```

响应：

```json
{
  "correlation_matrix": {
    "a": {
      "a": 1.0,
      "c": 1.0
    },
    "c": {
      "a": 1.0,
      "c": 1.0
    }
  }
}
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `csv_path` | str | **必需** | CSV 文件路径 |
| `columns` | list[str] | **必需** | 用于计算相关性的列名列表 |
| `method` | str | "pearson" | 相关性计算方法：`"pearson"`、`"spearman"` 或 `"kendall"` |

**相关性方法说明**：

- `pearson`: Pearson 相关系数，适用于线性关系
- `spearman`: Spearman 等级相关系数，适用于单调关系（不要求线性）
- `kendall`: Kendall Tau 相关系数，也适用于单调关系

### 创建微调任务

最小请求示例（必需包含 train_data_path, prediction_length, selected_groups）：

```bash
curl -X POST http://127.0.0.1:8011/v1/finetune/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "train_data_path": "/path/to/train.csv",
    "prediction_length": 96,
    "selected_groups": [
      {
        "target": "target",
        "covariates": []
      }
    ]
  }'
```

完整请求示例（包含所有可选参数）：

```bash
curl -X POST http://127.0.0.1:8011/v1/finetune/jobs \
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
    "selected_groups": [
      {
        "target": "target",
        "covariates": ["feature1", "feature2"]
      }
    ]
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
| `selected_groups` | list[object] | **必需** | 相关组列表，每个元素形如 `{"target": "...", "covariates": ["..."]}` |

> 每个 `selected_groups` 元素会独立训练一个模型。任务完成后会返回：
> - `output_dir`：该任务唯一结果目录
> - `target_model_map`：`target -> 模型目录` 显式映射
> - `model_paths`：兼容字段（由 `target_model_map` 派生）

**任务状态说明**：

- `queued`: 任务已创建，等待 worker 消费
- `running`: 后台 worker 正在执行 Chronos-2 微调
- `completed`: 训练成功完成
- `failed`: 训练过程中出现错误
- `cancelled`: 任务已取消（协作式取消）

### 支持的数据格式

当前版本的数据读取逻辑是：读取表格 → 按 `selected_groups` 中的列构造 Chronos 输入 →
每个 group 生成一个 `{"target": ..., "past_covariates": ...}` 的字典（列表传入 `fit()`）。
因此数据格式需要满足下面要求：

- 支持 CSV / Parquet
- 每一列被视为一个变量（variates），每一行是一个时间步
- 所选列必须是数值列（否则转换为 `float32` 会失败）
- `selected_groups` 中引用的列必须存在
- `target` 与 `covariates` 必须是数值列（否则转换为数值数组会失败）
- 当前版本不会解析 `item_id` / `timestamp` 等长表字段；若有这些列，请在上传前转换为“宽表”

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

当前版本已支持任务查询接口，包括详情和日志：

1) 查询任务详情

```bash
curl http://127.0.0.1:8011/v1/finetune/jobs/<job_id>
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
  "log_path": "./logs/550e8400-e29b-41d4-a716-446655440000.log",
  "model_paths": null,
  "target_model_map": null,
  "output_dir": "./artifacts/550e8400-e29b-41d4-a716-446655440000"
}
```

1) 查询任务日志（支持 tail）

```bash
# 返回完整日志
curl http://127.0.0.1:8011/v1/finetune/jobs/<job_id>/logs

# 返回最后 200 行
curl "http://127.0.0.1:8011/v1/finetune/jobs/<job_id>/logs?tail=200"
```

1) 任务列表

```bash
curl "http://127.0.0.1:8011/v1/finetune/jobs?limit=20"

# 仅查看排队中任务
curl "http://127.0.0.1:8011/v1/finetune/jobs?limit=20&status=queued"

# 仅查看进行中任务
curl "http://127.0.0.1:8011/v1/finetune/jobs?limit=20&status=running"

# 仅查看已完成任务
curl "http://127.0.0.1:8011/v1/finetune/jobs?limit=20&status=completed"
```

### 取消任务

当前版本支持**协作式取消**（cooperative cancellation），不会强制杀进程：

- `queued`：直接标记为 `cancelled`，任务不会被执行。
- `running`：设置 `cancel_requested=true`，训练回调/训练循环检测到后尽快中止，并最终标记为 `cancelled`。

**取消接口：**

```bash
curl -X POST http://127.0.0.1:8011/v1/finetune/jobs/<job_id>/cancel
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

`/v1/finetune/jobs/{job_id}` 会在详情中直接返回 loss 曲线（`metrics`）：

- `queued`：返回空字典 `{}`。
- `running` / `failed` / `cancelled`：返回当前已写入数据库的部分曲线。
- `completed`：返回完整曲线。

### 删除任务

1) 删除单个任务

```bash
curl -X DELETE http://127.0.0.1:8011/v1/finetune/jobs/<job_id>
```

说明：
- `queued` / `completed` / `failed` / `cancelled`：可直接删除（会清理任务目录与日志，不处理发布目录）。
- `running`：不允许直接删除，需先调用取消接口，待任务结束后再删除。

1) 批量删除任务（按状态）

```bash
curl -X DELETE "http://127.0.0.1:8011/v1/finetune/jobs?status=queued"
```

1) 批量删除任务（全部）

```bash
curl -X DELETE "http://127.0.0.1:8011/v1/finetune/jobs?all=true"
```

### 任务目录结构

任务创建后，会在 `artifacts/` 下生成以下结构：

```txt
artifacts/
└── <job_id>/
    ├── request.json          # 保存的请求参数
    ├── finetuned-ckpt_<target1>/  # 微调后的模型（训练完成后）
    │   ├── model.pt
    │   ├── config.json
    │   └── ...
    └── finetuned-ckpt_<target2>/  # 多组时会有多个目录
        └── ...
```

训练日志统一保存在 `LOGS_ROOT` 下：

```txt
logs/
└── <job_id>.log
```

## 配置

配置通过环境变量或 `.env` 文件管理：

```env
# 服务器设置
HOST=127.0.0.1
PORT=8011

# 数据库
SQLITE_DB_PATH=./data/finetune.db

# 路径
ARTIFACTS_ROOT=./artifacts
LOGS_ROOT=./logs
RELEASE_PATH=./release

# 规范兼容接口鉴权（为空表示不启用校验）
API_BEARER_TOKEN=

# 设备
DEVICE=cpu

# 训练
LOGGING_STEPS=100
FINETUNED_CKPT_NAME=finetuned-ckpt

```

> 训练使用的设备与日志频率、模型保存名称统一由配置控制，API 请求不再接收 `device` / `logging_steps` / `finetuned_ckpt_name` 参数。

## 目录结构

```text
ts_model_train_and_finetune/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用工厂 + 生命周期管理
│   │                         # 作用：创建FastAPI应用实例，管理应用生命周期（启动时初始化数据库、队列、worker；关闭时清理资源）
│   ├── api/                 # API 路由层
│   │   ├── __init__.py
│   │   ├── health.py        # 健康检查端点
│   │   │                     # 作用：提供简单的健康检查接口，用于监控服务状态
│   │   ├── finetune.py      # 微调任务相关端点
│   │   │                     # 作用：处理任务创建（POST /v1/finetune/jobs）、查询（GET /v1/finetune/jobs/{job_id}）、
│   │   │                     #      日志查询（GET /v1/finetune/jobs/{job_id}/logs）、
│   │   │                     #      任务取消（POST /v1/finetune/jobs/{job_id}/cancel）、
│   │   │                     #      模型发布（POST /v1/finetune/jobs/release）等API请求
│   │   └── tools.py         # 工具接口
│   │                         # 作用：提供数据分析工具，如相关性矩阵计算（POST /v1/tools/correlation）
│   ├── callbacks/           # 回调机制
│   │   ├── __init__.py
│   │   └── progress_callback.py  # 训练进度回调
│   │                         # 作用：在训练过程中实时更新数据库中的任务进度、损失值，并检查取消请求
│   ├── core/                # 核心配置和工具
│   │   ├── __init__.py
│   │   ├── config.py        # 设置管理
│   │   │                     # 作用：使用Pydantic管理应用配置，支持环境变量和.env文件
│   │   ├── paths.py         # 路径工具
│   │   │                     # 作用：提供路径解析和目录创建的工具函数
│   │   └── enums.py         # 状态枚举
│   │                         # 作用：定义任务状态（queued/running/completed/failed/cancelled）和微调模式枚举
│   ├── db/                  # 数据库层
│   │   ├── __init__.py
│   │   ├── session.py       # SQLAlchemy 会话设置
│   │   │                     # 作用：配置数据库连接和会话管理
│   │   ├── models.py        # ORM 模型
│   │   │                     # 作用：定义finetune_jobs表的SQLAlchemy模型
│   │   ├── crud.py          # CRUD 操作
│   │   │                     # 作用：提供数据库的创建、读取、更新、删除操作
│   │   └── init_db.py       # 数据库初始化
│   │                         # 作用：创建数据库表和初始数据
│   ├── schemas/             # 数据验证和响应模型
│   │   ├── __init__.py
│   │   ├── request.py       # 请求数据模型
│   │   │                     # 作用：定义API请求的数据结构和验证规则
│   │   └── response.py      # 响应数据模型
│   │                         # 作用：定义API响应的数据结构
│   ├── services/            # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── queue_service.py # 任务队列管理
│   │   │                     # 作用：管理内存中的任务队列，提供入队和出队操作
│   │   ├── job_service.py   # 任务业务逻辑
│   │   │                     # 作用：封装任务相关的业务操作，如状态更新、进度管理等
│   │   ├── trainer_service.py # 真实 Chronos-2 微调服务
│   │   │                     # 作用：调用Chronos-2官方API执行实际的模型微调训练
│   │   ├── dataset_service.py # 数据集加载与转换
│   │   │                     # 作用：加载CSV/Parquet数据文件，转换为Chronos-2所需的输入格式
│   │   └── model_service.py # 模型加载服务
│   │                         # 作用：加载本地缓存的Chronos-2基础模型
│   └── workers/             # 后台处理
│       ├── __init__.py
│       └── trainer_worker.py # 训练 worker 线程
│                         # 作用：后台线程持续从队列消费任务，调用训练服务执行微调
│
├── tests/
│   ├── __init__.py
│   ├── test_create_job.py    # 第 1 步测试（任务创建）
│   ├── test_worker_flow.py   # 第 2 步测试（异步工作流）
│   ├── test_trainer_service.py # 第 3 步测试（真实训练）
│   ├── test_query_api.py     # 第 4 步测试（查询接口）
│   └── test_cancel_job.py    # 第 5 步测试（任务取消）
│
├── pyproject.toml            # 项目依赖和配置
├── README.md                 # 项目文档
└── .gitignore               # Git忽略文件配置
```

### 请求处理流程

当收到一个微调任务创建请求时，系统按以下逻辑依次运行：

1. **HTTP请求接收** (`app/api/finetune.py`)
   - 接收POST `/v1/finetune/jobs` 请求
   - 使用Pydantic验证请求参数 (`app/schemas/request.py`)

2. **参数验证与预处理** (`app/api/finetune.py`)
   - 验证训练数据路径、预测长度等必需参数
   - 生成唯一任务ID (UUID)
   - 创建任务输出目录 (`app/core/paths.py`)

3. **数据库入库** (`app/db/crud.py`, `app/db/models.py`)
   - 在SQLite数据库中创建任务记录
   - 保存请求参数的JSON序列化
   - 设置初始状态为"queued"

4. **任务入队** (`app/services/queue_service.py`)
   - 将任务ID加入内存队列
   - 返回任务ID给客户端

5. **后台消费** (`app/workers/trainer_worker.py`)
   - Worker线程持续轮询队列
   - 获取任务ID后更新任务状态为"running"

6. **数据准备** (`app/services/dataset_service.py`)
   - 加载训练数据文件（CSV/Parquet）
   - 按 `selected_groups` 构造 Chronos-2 输入字典（`target` + `past_covariates`）

7. **模型训练** (`app/services/trainer_service.py`)
   - 加载Chronos-2基础模型 (`app/services/model_service.py`)
   - 调用官方`pipeline.fit()`方法进行微调
   - 通过回调机制实时更新进度 (`app/callbacks/progress_callback.py`)

8. **进度更新** (`app/callbacks/progress_callback.py`)
   - 每隔指定步数更新数据库中的current_step和last_loss
   - 每次拿到 loss 时写入 `finetune_job_losses` 曲线点（target, step, loss）
   - 检查取消请求，如有则抛出异常终止训练

9. **训练完成** (`app/services/trainer_service.py`)
   - 保存微调后的模型到输出目录
   - 更新任务状态为"completed"或"failed"

整个流程采用异步设计：API立即返回任务ID，实际训练在后台进行，支持并发请求和实时进度查询。

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
| model_paths | TEXT | 是 | NULL |
| target_model_map | TEXT | 是 | NULL |
| error_message | TEXT | 是 | NULL |
| current_step | INTEGER | 否 | 0 |
| max_steps | INTEGER | 否 | 0 |
| last_loss | FLOAT | 是 | NULL |
| cancel_requested | BOOLEAN | 否 | False |

> `target_model_map` 在数据库中以 JSON 字符串保存（对象形式：`target -> model_path`）。  
> `model_paths` 为兼容字段，以 JSON 字符串保存列表形式。

`finetune_job_losses` 表存储训练 loss 曲线点，字段如下：

| 字段 | 类型 | 可空 | 默认值 |
| ---- | ---- | ---- | ------ |
| id | INTEGER | 否 | 自增 |
| job_id | VARCHAR(36) | 否 | - |
| group_index | INTEGER | 否 | 0 |
| target | VARCHAR(255) | 否 | - |
| step | INTEGER | 否 | - |
| loss | FLOAT | 否 | - |
| created_at | DATETIME | 否 | 当前时间 |

约束与索引：

- 唯一约束：`(job_id, target, step)`（同 target 同一步会被更新覆盖，不会重复插入）
- 外键：`job_id -> finetune_jobs.id`
- 索引：`job_id`

## 前端 API 文档

### 基础信息

Base URL: `http://127.0.0.1:8011`  
Content-Type: `application/json`  
时间字段: ISO 8601（UTC）  
认证:
- 原始接口（`/v1/finetune/*`, `/v1/tools/*`, `/health`）默认无认证
- 兼容接口（`/api/v1/train_jobs*`, `/api/model/publish`, `/api/model/infer`）使用 Bearer Token（`API_BEARER_TOKEN` 非空时启用）

### 原始接口（保持兼容）

#### GET /health

用途: 健康检查  
响应 200:

```json
{
  "status": "ok"
}
```

#### POST /v1/finetune/jobs

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
| selected_groups | list[object] | - | 否 | 分组列表，元素形如 `{"target":"...","covariates":["..."]}` |

响应 201:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

#### GET /v1/finetune/jobs

用途: 查询最近任务列表  
查询参数:
- `limit` (int, 默认 20)
- `status` (str, 可选): `queued` / `running` / `completed`

#### GET /v1/finetune/jobs/{job_id}

用途: 查询任务详情、进度和 loss 曲线。  
`metrics` 字段说明：
- `queued`：`{}`
- `running` / `failed` / `cancelled`：返回当前可查询到的曲线点
- `completed`：返回完整曲线

补充说明：
- `metrics` 结构保持不变：`metrics.<target> = [loss...]`
- loss 来源改为数据库中显式 `target` 维度记录，不再依赖 `group_index` 回推
- 详情响应新增 `target_model_map` 与 `output_dir`，旧字段 `model_paths` 继续保留（兼容）

#### GET /v1/finetune/jobs/{job_id}/logs

用途: 查询日志（`text/plain`）  
查询参数: `tail` (int, 可选，仅返回最后 N 行)

#### POST /v1/finetune/jobs/{job_id}/cancel

用途: 请求取消任务（协作式取消）

#### DELETE /v1/finetune/jobs/{job_id}

用途: 删除单个任务。  
说明:
- `queued` / `completed` / `failed` / `cancelled`：可直接删除（会清理任务目录与日志，不处理发布目录）。
- `running`：不能直接删除，需先调用取消接口，等待状态变为非 `running` 后再删除。

成功响应 200:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted": true,
  "removed_from_queue": true,
  "files_deleted": 2,
  "message": "任务已删除"
}
```

失败响应示例:

```json
{
  "detail": "running 任务不能直接删除，请先取消任务后再删除"
}
```

#### DELETE /v1/finetune/jobs

用途: 批量删除任务。  
查询参数（二选一）:
- `status` (str): `queued` / `running` / `completed` / `failed` / `cancelled`
- `all` (bool): `true` 表示删除全部任务（`running` 会跳过）

成功响应 200:

```json
{
  "matched_jobs": 12,
  "deleted_jobs": 10,
  "skipped_running_jobs": 2,
  "removed_from_queue": 4,
  "files_deleted": 14,
  "message": "批量删除完成"
}
```

失败响应示例（参数冲突）:

```json
{
  "detail": "all=true 时不允许同时传 status"
}
```

失败响应示例（参数缺失）:

```json
{
  "detail": "请传 status 或 all=true"
}
```

#### POST /v1/finetune/jobs/release

用途: 发布已完成任务的模型目录。  
请求体:

| 字段 | 类型 | 必需 | 说明 |
| ---- | ---- | ---- | ---- |
| user_id | string | 是 | 用户 ID |
| job_id | string | 是 | 已训练任务 ID |
| version | string | 是 | 版本号 |

响应 200（成功）:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "/abs/path/to/release/u001_<job_id>_v1"
  }
}
```

说明:
- 目标发布目录已存在时，会先删除再复制（覆盖发布）。
- 当前接口返回的是发布目录绝对路径（字段名 `model_path`）。

### 兼容接口（规范对齐）

#### POST /api/v1/train_jobs

用途: 创建训练任务（规范响应包装）。  
鉴权: Bearer Token（配置启用时）

响应 200:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "job_id": "j_20260424_abc123"
  }
}
```

#### GET /api/v1/train_jobs/{job_id}

用途: 查询训练任务状态（规范字段）。  
鉴权: Bearer Token（配置启用时）

响应 200:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "job_id": "j_20260424_abc123",
    "is_completed": false,
    "status": "pending",
    "loss_data": {
      "steps": [1, 2],
      "values": [0.9, 0.7],
      "current_loss": 0.7
    },
    "duration": 128
  }
}
```

说明：
- 该接口继续遵循《接口文档规范说明》：外层固定 `code/message/data`。
- `loss_data` 字段结构保持不变（`steps/values/current_loss`），内部仅替换为基于 `target` 的库内聚合结果。

#### POST /api/model/publish

用途: 模型发布兼容接口（返回发布目录绝对路径）。  
鉴权: Bearer Token（配置启用时）

请求体:

```json
{
  "user_id": 10001,
  "version": "1.0.0",
  "job_id": "train_job_20260121103000"
}
```

成功响应 200:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000"
  }
}
```

失败示例（版本号格式非法）:

```json
{
  "code": 500,
  "message": "invalid version format, expected x.y.z",
  "data": null
}
```

说明:
- `model_path` 为发布目录绝对路径。
- 同一 `user_id + version + job_id` 多次调用，返回路径保持一致（覆盖发布）。

#### POST /api/model/infer

用途: 使用已发布模型进行推理预测。  
鉴权: Bearer Token（配置启用时）

请求体:

```json
{
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "cov_group": [
    {
      "target": "value1",
      "covariates": ["value2", "value3"]
    },
    {
      "target": "value2",
      "covariates": ["value1", "value4"]
    }
  ],
  "prediction_length": 3,
  "context_length": 64,
  "csv_path": "/abs/path/to/new_data.csv"
}
```

请求参数说明:

| 字段 | 类型 | 必需 | 说明 |
| ---- | ---- | ---- | ---- |
| `model_path` | str | 是 | 发布后的模型目录绝对路径 |
| `cov_group` | list[object] | 是 | 推理分组列表，元素形如 `{"target":"...","covariates":["..."]}` |
| `prediction_length` | int | 是 | 预测长度（正整数） |
| `context_length` | int | 是 | 上下文长度（正整数） |
| `csv_path` | str | 是 | 推理输入 CSV 路径 |

成功响应 200:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "predictions": [
      {
        "target": "value1",
        "prediction": [0.1, 0.2, 0.3]
      },
      {
        "target": "value2",
        "prediction": [0.4, 0.5, 0.6]
      }
    ]
  }
}
```

失败响应示例:

```json
{
  "code": 404,
  "message": "model path not found",
  "data": null
}
```

```json
{
  "code": 404,
  "message": "csv_path not found",
  "data": null
}
```

```json
{
  "code": 400,
  "message": "history length is insufficient",
  "data": null
}
```

```json
{
  "code": 404,
  "message": "model for target 'value1' not found",
  "data": null
}
```

说明:
- 推理时每个 `cov_group` 会按 `target` 选择对应子模型目录：`finetuned-ckpt_<target>`。
- 每个 `target` 使用滚动窗口推理：每次取 `context_length` 行作为输入，预测接下来 `prediction_length` 个点，窗口按 `prediction_length` 前进。
- 每个 `target` 的最终输出长度为 `n - context_length`（`n` 为 CSV 总行数）；最后一个窗口会按剩余长度裁剪。
- 一次请求包含多个 `target` 时，会逐组推理并按 `cov_group` 顺序返回结果。

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

- ✅ 任务查询端点（详情 / 日志）
- ✅ 详情接口返回按预测目标组织的 loss 曲线（`metrics.<target> = [loss...]`）

**第 5 步（已完成）**：取消与辅助接口

- ✅ 任务取消接口（协作式取消）

## 许可证

内部项目，仅供模型微调研究使用。
