# 训练任务 target-model 显式关联改造 TODO

## 结论（先看）

你的 3 点要求能够保证后续按 `target` 正确使用模型，方向是可行的。

但要完全落地，需要补充一个关键点：
- `loss` 记录也要在库里显式带 `target`（或单独 target 维度表），否则查询仍依赖 `group_index -> request_json.selected_groups` 的隐式映射。

---

## 目标状态

1. 单任务产物仍是一个目录（`job.output_dir`），目录内可包含多个子模型目录。  
2. 数据库存储 `target -> model_path` 显式映射（而非仅 `model_paths` 列表）。  
3. 查询 loss 直接按数据库中 `target` 聚合返回，不再通过 `group_index` 回推。  
4. 训练完成返回任务结果目录（单目录），发布接口复制该目录（单目录复制）。  

---

## 必改项（代码）

1. 数据库模型：任务模型增加/替换字段
- 在 `finetune_jobs` 增加 `target_model_map`（TEXT，JSON 字符串）：
  - 结构建议：`{"target_a": "/abs/.../finetuned-ckpt_target_a", "target_b": "..."}`
- 保留 `model_paths` 作为兼容字段（短期），新逻辑不再依赖它。

2. 数据库模型：loss 表增加 target 维度
- 在 `finetune_job_losses` 增加 `target`（STRING/TEXT, NOT NULL）。
- 唯一键改为 `(job_id, target, step)`。
- `group_index` 可保留兼容（短期可写入但不参与查询核心逻辑）。

3. 训练服务返回结构改造
- `train_chronos2()` 返回从 `list[str]` 改为 `dict[str, str]`（`target -> model_path`）。
- `trainer_worker` / `job_service.complete_job_training` / `crud.mark_job_completed` 全链路改为存映射。

4. 训练回调写 loss 改造
- `upsert_job_loss_point()` 入参增加 `target`，写入 `finetune_job_losses.target`。
- 回调层在每次写 loss 时使用当前组 `target`（已有 `active_group_target`，可直接复用）。

5. 查询接口改造
- `job_service._build_loss_metrics()` 改为：
  - 直接从 `finetune_job_losses` 按 `target, step` 查询并组装 `metrics[target] = [loss...]`。
  - 移除对 `request_json` 与 `_extract_group_targets` 的依赖。
 - 对外响应格式保持不变：
   - `/v1/finetune/jobs/{job_id}` 仍返回现有 `metrics` 结构（`dict[str, list[float]]`）。
   - `/api/v1/train_jobs/{job_id}` 仍返回现有 `loss_data` 结构（`steps/values/current_loss`）。
   - 仅替换内部数据来源与转换逻辑，不新增/删除接口字段。

6. 任务详情返回字段调整
- `JobDetailResponse` 建议新增：
  - `output_dir: str`
  - `target_model_map: dict[str, str] | null`
- `model_paths` 标记为兼容字段（可选保留一段时间）。

7. 发布逻辑确认（两条接口）
- `/v1/finetune/jobs/release`：继续复制 `job.output_dir`（单目录），返回发布目录。
- `/api/model/publish`：继续复制 `job.output_dir` 到目标发布目录；当前响应 `model_path` 命名为文件路径语义（`.../model.bin`），建议评估是否改为目录语义或补充说明。

---

## 迁移与兼容（必须有）

1. 初始化迁移脚本（`init_db`）
- 增加 `target_model_map` 列检测与补列。
- `finetune_job_losses` 表结构迁移（加 `target` 与新唯一键）。

2. 旧数据回填策略
- `target_model_map` 回填：
  - 优先从 `request_json.selected_groups` + `model_paths` 按旧顺序 zip 回填（仅历史兼容）。
- `loss.target` 回填：
  - 同理用历史 `group_index` 与 `selected_groups` 对应回填。
- 回填失败（历史脏数据）时记录告警并降级为 `group_{n}`。

3. 对外兼容策略
- 过渡期仍返回 `model_paths`（由 `target_model_map.values()` 派生）避免前端立即改动。
- 文档标注 `model_paths` 将废弃，推荐使用 `target_model_map + output_dir`。
- loss 兼容策略（接口格式不变）：
  - 旧逻辑：`group_index -> target` 映射后再拼响应。
  - 新逻辑：直接按库中 `target` 聚合，再映射到原有响应结构。
  - 若 `/api/v1/train_jobs/{job_id}` 当前只返回单条曲线，保持该行为不变；目标选择规则写死为“第一个 target（按 selected_groups 顺序）”或“指定默认 target”，并在文档中明确。

---

## 文档与测试

1. README/API 文档更新
- 任务详情响应示例改为以 `target_model_map` 为主。
- 明确“训练结果与发布单位是任务目录，不是模型路径列表”。
- 明确 loss 来源为库内 `target` 维度数据。

2. 测试用例新增/修改
- 创建多 target 任务后，断言库内存在正确 `target_model_map`。
- 断言 `/jobs/{id}` 的 `metrics` 不依赖 `request_json` 也可返回正确结果。
- 断言发布接口复制的是单目录且目录内含多个 target 子模型。
- 迁移测试：旧库升级后可查询历史任务。

---

## 验收标准（Done Definition）

- 多 target 任务训练完成后，数据库可直接读到 `target -> model_path`。
- 任意一个 target 的 loss 曲线查询不依赖 `group_index` 推断。
- 两个发布接口都以“任务目录”为复制单位，且返回的是发布目录信息。
- 旧数据可读，旧接口不崩溃，兼容期行为可预期。
