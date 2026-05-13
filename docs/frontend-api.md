# 前端 API 文档

## 基础信息

- Base URL: `http://127.0.0.1:8011`
- Content-Type: `application/json`
- 时间字段: ISO 8601（UTC）
- 认证：
- 原始接口（`/v1/finetune/*`, `/v1/tools/*`, `/health`）默认无认证
- 兼容接口（`/api/v1/train_jobs*`, `/api/model/*`）在 `API_BEARER_TOKEN` 非空时启用 Bearer Token 校验

## 原始接口

### GET /health

用途：健康检查

响应：

```json
{
  "status": "ok"
}
```

### POST /v1/finetune/jobs

用途：创建训练任务

请求体：

| 字段 | 类型 | 默认值 | 必需 | 说明 |
| ---- | ---- | ------ | ---- | ---- |
| train_data_path | str | - | 是 | 训练数据路径（CSV/Parquet） |
| val_data_path | str \| null | null | 否 | 验证数据路径 |
| prediction_length | int | - | 是 | 预测长度（正整数） |
| context_length | int | 512 | 否 | 上下文长度（正整数） |
| finetune_mode | str | lora | 否 | `lora` 或 `full` |
| learning_rate | float | 1e-4 | 否 | 学习率（正数） |
| num_steps | int | 1000 | 否 | 训练步数（正整数） |
| batch_size | int | 32 | 否 | 批大小（正整数） |
| selected_groups | list[object] | - | 是 | 形如 `{"target":"...","covariates":["..."]}` |

响应：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

### GET /v1/finetune/jobs

用途：查询任务列表

查询参数：

- `limit` (int, 默认 20)
- `status` (str, 可选): `queued` / `running` / `completed` / `failed` / `cancelled`

### GET /v1/finetune/jobs/{job_id}

用途：查询任务详情、进度、loss 曲线

说明：

- `metrics.<target> = [loss...]`
- `queued` 返回 `{}`
- `running/failed/cancelled` 返回当前可用曲线
- `completed` 返回完整曲线
- 响应含 `target_model_map` 和 `output_dir`；`model_paths` 保留兼容

### GET /v1/finetune/jobs/{job_id}/logs

用途：查询日志（`text/plain`）

查询参数：

- `tail` (int, 可选): 返回最后 N 行

### POST /v1/finetune/jobs/{job_id}/cancel

用途：协作式取消任务

### DELETE /v1/finetune/jobs/{job_id}

用途：删除单个任务

说明：

- `queued/completed/failed/cancelled` 可直接删除
- `running` 会先内部取消并等待退出 running；超时返回冲突

成功响应：

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted": true,
  "removed_from_queue": true,
  "files_deleted": 2,
  "message": "任务已删除"
}
```

### DELETE /v1/finetune/jobs

用途：批量删除任务

查询参数（二选一）：

- `status`：按状态删除
- `all=true`：删除全部（`running` 尝试内部取消，超时跳过）

### POST /v1/finetune/jobs/release

用途：发布已完成任务模型

请求体：

| 字段 | 类型 | 必需 | 说明 |
| ---- | ---- | ---- | ---- |
| user_id | string | 是 | 用户 ID |
| job_id | string | 是 | 训练任务 ID |
| version | string | 是 | 版本号 |

响应（成功）：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "/abs/path/to/release/u001_<job_id>_v1"
  }
}
```

## 兼容接口（规范包装）

### POST /api/v1/train_jobs

用途：创建任务（`code/message/data`）

### GET /api/v1/train_jobs/{job_id}

用途：查询任务状态（`code/message/data`）

说明：`loss_data` 保持 `steps/values/current_loss` 结构。

### POST /api/model/publish

用途：发布模型兼容接口

说明：同一 `user_id + version + job_id` 重复调用为覆盖发布。

### POST /api/model/infer

用途：使用发布模型推理

简化请求体：

```json
{
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "csv_path": "/abs/path/to/new_data.csv"
}
```

高级请求体支持覆盖：`cov_group`、`prediction_length`、`context_length`。

输出核心字段：

- `data.predictions[].target`
- `data.predictions[].prediction`
- `data.predictions[].actual`

### GET /api/model/infer/config

用途：查询模型默认推理参数（来自 `metadata.json`）

查询参数：

- `model_path` (str, 必需，绝对路径)

成功响应示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
    "prediction_length": 96,
    "context_length": 512
  }
}
```

常见错误：

- `code=404, message="model path not found"`
- `code=404, message="metadata.json not found"`
- `code=400, message="metadata.json missing required field: ..."`

### POST /api/model/infer/chunk

用途：分段推理（按 `task_id` 复用模型，末段释放）

请求体：

| 字段 | 类型 | 必需 | 说明 |
| ---- | ---- | ---- | ---- |
| task_id | str | 是 | 分段推理任务 ID |
| model_path | str | 是 | 发布模型绝对路径 |
| is_last_segment | bool | 是 | 是否最后一段 |
| segment | list[object] | 是 | 行对象数组，每个对象表示 CSV 一行 |

请求示例：

```json
{
  "task_id": "infer_task_20260512_001",
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "is_last_segment": false,
  "segment": [
    {"time": 47, "value1": 0.0121, "value2": 0.0173, "value3": 0.0113, "value4": 0.0265},
    {"time": 48, "value1": 0.0579, "value2": 0.0140, "value3": 0.0338, "value4": 0.0302},
    {"time": 49, "value1": 0.0610, "value2": 0.0140, "value3": 0.0340, "value4": 0.0310},
    {"time": 50, "value1": 0.0620, "value2": 0.0150, "value3": 0.0350, "value4": 0.0320}
  ]
}
```

成功响应示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "infer_task_20260512_001",
    "predictions": [
      {
        "target": "value1",
        "prediction": [0.1, 0.2],
        "actual": [0.061, 0.062]
      }
    ],
    "model_reused": false,
    "released": false
  }
}
```

说明：

- `data.predictions` 与 `/api/model/infer` 的结构一致。
- 同一 `task_id` 重复调用会复用缓存模型；当 `is_last_segment=true` 时释放缓存。
- 若同一 `task_id` 携带不同 `model_path`，返回 `code=409`。

### GET /api/model/info

用途：查询发布模型元数据（targets、selected_groups、prediction/context length）

## 工具接口

### POST /v1/tools/correlation

用途：计算相关性矩阵

请求示例：

```json
{
  "csv_path": "/path/to/data.csv",
  "columns": ["a", "c"],
  "method": "pearson"
}
```

`method` 支持：`pearson` / `spearman` / `kendall`
