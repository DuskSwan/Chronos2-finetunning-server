# 推理接口支持数据片段 TODO（分段推理）

## 1. 需求确认

目标：支持前端“按小段数据多次调用推理”，后端按任务维度复用模型，避免重复加载。

新增两个接口：

1. 模型参数查询接口
- 前端传 `model_path`（模型绝对路径）
- 后端读取该模型训练元数据，返回：
  - `prediction_length`
  - `context_length`

2. 分段推理接口
- 前端传：
  - 一小段数据（本次数据段）
  - `task_id`（本次预测任务ID）
  - `model_path`（模型绝对路径）
  - `is_last_segment`（是否末尾段）
- 后端行为：
  - 若 `task_id` 是新任务：加载模型并缓存
  - 若 `task_id` 已存在：复用缓存中的模型
  - 对本段数据执行一次推理并返回本次结果
  - 若 `is_last_segment=true`：本次推理结束后释放该 `task_id` 对应模型缓存

---

## 2. 数据段格式约定（已确认）

前端传入的 `segment` 是“行对象数组”，每个字典表示 CSV 中的一行。

示例：

```json
[
  {
    "time": 47,
    "value1": 0.012157337,
    "value2": 0.017334247,
    "value3": 0.011365411,
    "value4": 0.02653957
  },
  {
    "time": 48,
    "value1": 0.057906676,
    "value2": 0.013979295,
    "value3": 0.033833418,
    "value4": 0.030267294
  }
]
```

约束建议：
- `segment` 必须是非空数组。
- 每个元素必须是对象（dict）。
- 列名语义与训练/推理列名保持一致。
- `time` 作为时间列可保留参与排序/对齐，但不作为 target/covariate 数值特征（按当前训练规则）。

---

## 3. 设计原则

- `task_id` 是模型缓存复用的主键。
- 同一 `task_id` 在一个任务周期内应绑定固定的 `model_path`。
- 末段释放必须在响应前或响应后立即执行（推荐 `finally` 里兜底释放）。
- 保持现有返回结构风格（`code/message/data`）一致。
- 错误信息要明确区分：请求错误 / 模型不存在 / 元数据缺失 / 推理失败。

---

## 4. 接口草案

### 4.1 模型参数查询接口

建议：`GET /api/model/infer/config`

请求参数：
- `model_path`: string, required

响应示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "/abs/path/to/model",
    "prediction_length": 96,
    "context_length": 512
  }
}
```

校验与错误：
- `model_path` 为空 -> 400
- 路径不存在 -> 404
- `metadata.json` 缺失或字段缺失 -> 400/404（按现有规范）

### 4.2 分段推理接口

建议：`POST /api/model/infer/chunk`

请求体示例：

```json
{
  "task_id": "infer_task_20260512_001",
  "model_path": "/abs/path/to/model",
  "is_last_segment": false,
  "segment": [
    {
      "time": 47,
      "value1": 0.012157337,
      "value2": 0.017334247,
      "value3": 0.011365411,
      "value4": 0.02653957
    },
    {
      "time": 48,
      "value1": 0.057906676,
      "value2": 0.013979295,
      "value3": 0.033833418,
      "value4": 0.030267294
    }
  ]
}
```

响应示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "infer_task_20260512_001",
    "predictions": [
      {
        "target": "value1", // 预测目标
        "prediction": [0.1, 0.2, 0.3], // 预测值
        "actual": [0.05, 0.25, 0.35], // 真实值
      }
    ]
  }
}
```

说明：
- `data.predictions` 结构与现有 `/api/model/infer` 保持一致。
- 兼容核心字段不变：`target`、`prediction`、`actual`。
- 分段接口仅额外增加任务态字段：`task_id`。

---

## 5. 核心实现点 TODO

### 5.1 Schema 定义

- [x] 新增“模型参数查询”请求/响应 Schema。
- [x] 新增“分段推理”请求/响应 Schema。
- [ ] 校验字段：
  - [x] `task_id` 非空
  - [x] `model_path` 为绝对路径且非空
  - [x] `segment` 非空数组
  - [x] `segment[*]` 为对象（每项表示一行）
  - [x] `is_last_segment` 必填布尔

### 5.2 路由与接口

- [x] 新增 `GET /api/model/infer/config` 路由。
- [x] 新增 `POST /api/model/infer/chunk` 路由。
- [x] 接口统一返回 `code/message/data`。

### 5.3 模型元数据读取复用

- [x] 复用现有 metadata 读取工具（如 `model_metadata_service`）。
- [x] 从 metadata 提取 `prediction_length/context_length`。
- [x] 缺失字段时返回清晰错误。

### 5.4 任务级模型缓存管理（重点）

- [x] 新增进程内缓存容器：`task_id -> {model_path, model_instance, last_access_at}`。
- [x] 新任务：加载模型并写入缓存。
- [x] 老任务：直接复用缓存模型。
- [ ] 防御性校验：
  - [x] 若相同 `task_id` 传入不同 `model_path`，返回冲突错误（建议 409）。
- [x] `is_last_segment=true` 时释放缓存：
  - [x] 删除缓存引用
  - [ ] 执行显存/内存清理（按框架支持）
- [x] 异常兜底：末段即使推理失败，也应尝试释放。

### 5.5 并发与线程安全

- [x] 为缓存操作加锁（至少写操作加锁）。
- [x] 避免同一 `task_id` 并发触发重复加载。
- [ ] 明确服务多进程部署限制：
  - [ ] 若多 worker，每个 worker 缓存独立（文档说明）。

### 5.6 生命周期与泄漏防护

- [x] 增加可选 TTL 清理机制（防止前端漏传末段导致泄漏）。
- [x] 增加最大缓存任务数上限（防 OOM）。
- [x] 达到上限时给出清晰错误或淘汰策略（建议先报错，后续再LRU）。

### 5.7 分段数据校验与组装

- [x] 将 `segment` 转为 DataFrame（或当前推理输入结构）。
- [x] 校验 `segment` 中包含模型所需列（target/covariates）。
- [x] 非数值列返回清晰错误。

### 5.8 推理服务适配

- [x] 提取“单次分段推理”服务函数，供新接口调用。
- [x] 与现有 `/api/model/infer` 全量推理逻辑解耦，避免互相回归。
- [x] 返回结构对齐 `/api/model/infer`：`data.predictions[].target/prediction/actual`。
- [x] 分段接口可额外返回任务态字段：`task_id`、`model_reused`、`released`。

### 5.9 观测与日志

- [ ] 记录关键日志：
  - [x] `task_id` 首次加载
  - [x] `task_id` 复用命中
  - [x] `task_id` 末段释放
  - [x] 缓存当前大小
- [ ] 增加异常日志，便于定位泄漏和并发问题。

---

## 6. 错误码与异常场景清单

- [x] `task_id` 缺失/为空 -> 400
- [x] `model_path` 非法或不存在 -> 404/400
- [x] metadata 缺失 `prediction_length/context_length` -> 400
- [x] 同一 `task_id` 切换 `model_path` -> 409
- [x] `segment` 非数组或为空 -> 400
- [x] `segment` 缺失所需列 -> 400
- [x] `segment` 列类型错误 -> 400
- [ ] 模型加载失败 -> 500
- [ ] 推理执行失败 -> 500

---

## 7. 测试用例 TODO

### 7.1 模型参数查询接口

- [x] 正常返回 `prediction_length/context_length`。
- [x] `model_path` 不存在时返回错误。
- [x] metadata 缺字段时报错。

### 7.2 分段推理接口功能

- [x] 首段请求触发模型加载。
- [x] 非首段同 `task_id` 复用模型（不重复加载）。
- [x] 末段请求完成后释放缓存。
- [x] 末段后再次同 `task_id` 请求应重新加载。

### 7.3 一致性与防错

- [x] 同 `task_id` + 不同 `model_path` 返回冲突。
- [ ] 推理异常时，若 `is_last_segment=true` 仍释放缓存。
- [ ] 高并发下同 `task_id` 不重复加载（或重复加载次数可控并有保护）。

### 7.4 分段数据格式

- [x] `segment` 为行对象数组时可正常推理。
- [x] `segment` 行中缺列时返回 400。
- [x] `segment` 行中出现非数值特征列时报错。
- [ ] `segment` 含 `time` 列时排序逻辑正确。

### 7.5 泄漏防护

- [x] 未发送末段时 TTL 能自动回收。
- [x] 缓存上限触发行为符合预期。

---

## 8. 实施顺序建议

1. [ ] 定义新接口 Schema 与路由。
2. [x] 完成 metadata 查询接口。
3. [x] 实现 `task_id` 缓存管理器（含加锁、加载、释放）。
4. [x] 接入分段推理主流程。
5. [ ] 补齐错误码和日志。
6. [ ] 编写单元测试与集成测试。
7. [ ] 更新 `README.md` 与 `docs/frontend-api.md`。

---

## 9. 前后端协作约定（需确认）

- [ ] `task_id` 由前端生成且全局唯一（至少在单次业务周期内唯一）。
- [ ] 前端必须在最后一段设置 `is_last_segment=true`。
- [ ] 若前端中断，后端依赖 TTL 回收。
- [ ] `segment` 已确认采用“行对象数组”（每个对象=CSV一行）。
