# 后端架构与开发说明

## 架构概览

```text
HTTP 请求
    ↓
FastAPI 路由
    ↓
参数验证 → 数据库入库 → 入队
    ↓
后台 Worker
    ↓
加载数据 → Chronos-2 fit()
    ↓
Callback 更新进度 → 保存模型 → 更新状态
```

技术栈：

- API: FastAPI + Uvicorn
- ORM/DB: SQLAlchemy + SQLite
- 队列: `queue.Queue`
- 模型: `chronos-forecasting`
- 数据: Pandas + PyArrow

## 请求处理流程（训练任务）

1. 接收 `POST /v1/finetune/jobs` 请求并做参数验证。
2. 生成 `job_id`，创建任务目录，写入 `request.json`。
3. 入库为 `queued` 状态。
4. 任务 `job_id` 进入内存队列。
5. worker 消费任务并置为 `running`。
6. 加载 CSV/Parquet，按 `selected_groups` 组装模型输入。
7. 调用 Chronos-2 `fit()` 执行训练。
8. callback 回写 `current_step/max_steps/last_loss` 与 loss 曲线。
9. 训练结束后置为 `completed/failed/cancelled`。

## 任务状态机

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

取消为协作式：`running` 状态下通过 `cancel_requested=true` 触发中止。

## 数据格式要求

- 支持 CSV / Parquet
- 宽表：每列一个变量，每行一个时间步
- `selected_groups` 中引用列必须存在且可转数值
- 当前不解析长表 `item_id/timestamp` 语义，需预先转宽表

## 关键目录

```text
app/
├── api/          # 路由
├── callbacks/    # 训练回调
├── core/         # 配置/路径/枚举
├── db/           # session/model/crud/init
├── schemas/      # 请求/响应模型
├── services/     # queue/job/dataset/trainer/model
└── workers/      # trainer worker
```

## 任务产物

```text
artifacts/
└── <job_id>/
    ├── request.json
    ├── finetuned-ckpt_<target1>/
    └── finetuned-ckpt_<target2>/

logs/
└── <job_id>.log
```

## 数据库结构

### finetune_jobs

核心字段：

- `id`, `status`, `request_json`
- `created_at`, `started_at`, `finished_at`
- `output_dir`, `log_path`
- `model_paths`, `target_model_map`
- `current_step`, `max_steps`, `last_loss`
- `error_message`, `cancel_requested`

### finetune_job_losses

字段：

- `id`, `job_id`, `group_index`, `target`, `step`, `loss`, `created_at`

约束：

- 唯一约束：`(job_id, target, step)`
- 外键：`job_id -> finetune_jobs.id`

## 配置项

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

说明：`device/logging_steps/finetuned_ckpt_name` 由配置统一控制，不从训练 API 请求体读取。

## 测试

```bash
pytest tests/ -v
pytest tests/ --cov=app
```

常用模块：

- `tests/test_create_job.py`
- `tests/test_worker_flow.py`
- `tests/test_trainer_service.py`
- `tests/test_query_api.py`
- `tests/test_cancel_job.py`
