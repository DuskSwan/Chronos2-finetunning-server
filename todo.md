# 训练任务 target-model 显式关联改造 TODO（实施版）

## 0. 约束与边界

1. 接口兼容约束
- 旧接口路径、请求参数、响应主结构不变。
- `/api/*` 兼容接口必须继续遵循《接口文档规范说明.md》：
  - 外层结构保持 `code/message/data`
  - 字段命名保持 `snake_case`
  - 既有业务字段语义不变

2. 变更承载方式
- 涉及接口返回信息扩展时，只能通过“新增变量/新增字段”承载，不改旧字段语义。
- 旧字段不删除、不重命名、不改类型（至少在兼容期内）。

3. 发布与产物约束
- 单个任务的训练结果仍以 `job.output_dir` 为唯一结果目录。
- 发布接口复制单位是“任务目录”，不是“模型路径列表”。

## 1. 数据模型改造

1. `finetune_jobs` 新增字段
- 新增 `target_model_map`（TEXT，可空，JSON 字符串）。
- 结构：`{"<target>": "<abs_model_dir_path>", ...}`。
- 保留原 `model_paths` 字段用于兼容读取。

2. `finetune_job_losses` 新增字段
- 新增 `target`（STRING/TEXT，非空）。
- 唯一约束调整为 `(job_id, target, step)`。
- `group_index` 保留兼容（迁移期可继续写入，不作为查询主键维度）。

3. 初始化与迁移脚本
- 在 `app/db/init_db.py` 增加列探测与补列逻辑：
  - `finetune_jobs.target_model_map`
  - `finetune_job_losses.target`
- 处理唯一约束变更（必要时重建 `finetune_job_losses` 表）。

## 2. 训练链路改造

1. 训练结果返回结构
- `app/services/trainer_service.py::train_chronos2()` 返回值从 `list[str]` 改为 `dict[str, str]`（`target -> model_path`）。
- 仍按当前目录规则保存子模型：`<output_dir>/<finetuned_ckpt_name>_<target>/...`。

2. worker 与任务完成入库
- `app/workers/trainer_worker.py` 接收新的映射返回值。
- `app/services/job_service.py::complete_job_training()` 改为接收并写入 `target_model_map`。
- `app/db/crud.py::mark_job_completed()` 增加 `target_model_map` 持久化。
- 兼容写入：可由 `target_model_map.values()` 派生旧 `model_paths`（仅兼容用途）。

3. 训练过程 loss 入库
- `app/db/crud.py::upsert_job_loss_point()` 入参新增 `target`。
- 回调中每次写 loss 时传入当前 `active_group_target`。
- 数据库存储层不再依赖 `group_index -> target` 推断。

## 3. 查询与响应适配（接口格式不变）

1. 内部查询逻辑改造
- `app/services/job_service.py::_build_loss_metrics()` 改为直接按数据库 `target, step` 聚合。
- 删除对 `request_json.selected_groups` 的 target 回推依赖。

2. `/v1/finetune/jobs/{job_id}` 保持兼容
- 保持现有响应字段不删：
  - `model_paths` 继续返回（兼容）
  - `metrics` 结构保持 `dict[str, list[float]]`
- 可新增字段（新增变量方式）：
  - `target_model_map`（推荐新增）
  - `output_dir`（如当前未返回可新增）

3. `/api/v1/train_jobs/{job_id}` 保持规范
- 外层 `code/message/data` 不变。
- `loss_data` 结构保持 `steps/values/current_loss` 不变。
- 若当前实现是单曲线输出，保持该行为；新增内部规则变量用于确定默认 target：
  - 示例变量：`default_loss_target_strategy`（如 `first_selected_target`）
  - 仅影响内部选择逻辑，不改变响应格式。

## 4. 发布链路确认

1. `/v1/finetune/jobs/release`
- 继续复制 `job.output_dir` 到发布目录。
- 返回值结构不变（旧字段保持）。

2. `/api/model/publish`
- 继续复制 `job.output_dir` 到规范发布目录。
- 返回结构继续保持 `code/message/data.model_path`。
- 如需补充目录语义，仅通过新增变量/文档说明，不改 `model_path` 字段。

## 5. 旧数据兼容与回填

1. `target_model_map` 回填
- 从历史 `request_json.selected_groups` 与 `model_paths` 按顺序 zip 生成。
- 回填失败时降级 target 名为 `group_<n>` 并记录日志。

2. `finetune_job_losses.target` 回填
- 从历史 `group_index` 对应 `selected_groups[target]` 回填。
- 回填失败时降级 `group_<n>` 并记录日志。

3. 读取兼容顺序
- 优先读新字段：`target_model_map` / `loss.target`。
- 新字段缺失时，按旧字段兜底并在日志中告警。

## 6. 文档更新

1. 更新 `README.md`
- 明确“任务结果是目录级单位”。
- 明确“数据库显式存储 `target -> model_path`”。
- 明确“loss 按 `target` 入库，接口格式不变”。

2. 更新接口说明（含规范对齐）
- 标注兼容字段与新增字段关系：
  - `model_paths`（兼容）
  - `target_model_map`（新增）
- `/api/*` 文档继续严格遵循《接口文档规范说明.md》。

## 7. 测试计划

1. 单元测试
- `crud`：`target_model_map` 序列化/反序列化、`loss.target` upsert 唯一键。
- `job_service`：不依赖 `request_json` 仍可构建 `metrics`。

2. 集成测试
- 多 target 任务完成后：
  - `target_model_map` 正确
  - `metrics` 按 target 返回正确曲线
  - `model_paths` 兼容字段仍可用
- 两个发布接口均复制单目录并成功返回。

3. 迁移测试
- 旧库升级后可查询旧任务、可发布旧任务、接口不报错。
- `/api/v1/train_jobs*` 响应结构完全匹配规范。

## 8. 交付验收清单

1. 数据层
- 可直接查询到 `target_model_map` 与 `loss.target`。

2. 服务层
- 训练完成入库不再依赖隐式 `group_index -> target` 映射。

3. 接口层
- 旧接口字段与语义保持不变。
- 新信息仅通过新增变量/新增字段提供。
- `/api/*` 响应保持 `code/message/data` 且字段命名规范不变。

4. 发布层
- 发布复制单位为任务目录，非模型列表。

5. 兼容性
- 历史数据可读可查可发布，行为可预期。
