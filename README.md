# Chronos-2 模型微调服务

基于 FastAPI 的 Chronos-2 时间序列模型训练与推理服务，支持 LoRA/全量微调、任务队列、模型发布，以及边端离线推理。

## 项目定位

本项目面向两类场景：

- 服务端训练：通过 HTTP API 提交、查询、取消微调任务。
- 边端离线推理：将训练后模型以目录/二进制方式交付，在离线环境执行推理。

## 核心能力

- 异步训练任务队列（SQLite 持久化 + 后台 worker）
- 真实 Chronos-2 微调（`fit()`）与训练进度跟踪
- 任务查询、日志查询、协作式取消、任务删除
- 模型发布与兼容接口（`/api/v1/train_jobs*`, `/api/model/*`）
- 边端离线推理 CLI 与 ExternalNode(ZMQ REQ) 长驻进程

## 文档导航

- 前端 API 文档：[`docs/frontend-api.md`](docs/frontend-api.md)
- 边端离线推理 CLI（含二进制包）：[`docs/edge-offline-cli.md`](docs/edge-offline-cli.md)
- 后端架构与开发说明：[`docs/backend-architecture.md`](docs/backend-architecture.md)

## 快速开始

### 1. 环境要求

- Python 3.11+
- pip（或 uv）

### 2. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 方式 A
uvicorn app.main:app --host 127.0.0.1 --port 8011

# 方式 B
python -m uvicorn app.main:app --host 127.0.0.1 --port 8011 --reload

# 方式 C（读取 .env 的 HOST/PORT）
python -m app.main
```

### 4. 健康检查

```bash
curl http://127.0.0.1:8011/health
```

## 目录结构（简版）

```text
ts_model_train_and_finetune/
├── app/          # 服务代码（API / 训练 / 推理 / worker）
├── docs/         # 文档
├── scripts/      # 打包与辅助脚本
├── tests/        # 测试
├── README.md
└── pyproject.toml
```

## 配置

通过环境变量或 `.env` 管理：

```env
HOST=127.0.0.1
PORT=8011
SQLITE_DB_PATH=./data/finetune.db
ARTIFACTS_ROOT=./artifacts
LOGS_ROOT=./logs
RELEASE_PATH=./release
API_BEARER_TOKEN=
DEVICE=cpu
LOGGING_STEPS=100
FINETUNED_CKPT_NAME=finetuned-ckpt
```

## 测试

```bash
pytest tests/ -v
pytest tests/ --cov=app
```

## 许可证

内部项目，仅供模型微调研究使用。
