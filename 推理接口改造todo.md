# 推理接口改造 TODO：让模型自动携带训练时的相关组配置

## 1. 改造背景

当前 `/api/model/infer` 推理接口要求前端在每次推理时传入 `cov_group`，例如：

```json
{
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "cov_group": [
    {
      "target": "value1",
      "covariates": ["value2", "value3"]
    }
  ],
  "prediction_length": 3,
  "context_length": 64,
  "csv_path": "/abs/path/to/new_data.csv"
}
```

这个设计对用户不友好，因为用户需要记住训练模型时：

- 哪些变量被作为 `target`
- 每个 `target` 分别绑定了哪些 `covariates`
- 推理时应该使用哪个 `context_length`
- 推理时应该使用哪个 `prediction_length`

实际上，这些信息在训练任务创建时已经存在于 `selected_groups` 和训练参数中，应该被视为模型的一部分，而不应该让用户在推理时再次手动输入。

---

## 2. 改造目标

本次改造的核心目标是：

> 让模型发布物携带训练时的元数据，推理时默认从模型目录读取配置，前端不再强制传入 `cov_group`。

改造后，普通用户只需要传入：

```json
{
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "csv_path": "/abs/path/to/new_data.csv"
}
```

后端自动从：

```text
<model_path>/metadata.json
```

读取：

- `selected_groups`
- `target`
- `covariates`
- `prediction_length`
- `context_length`
- 子模型目录映射关系

然后自动完成推理。

---

## 3. 总体设计思路

### 3.1 训练配置应随模型发布

训练任务创建时，用户已经传入了类似这样的参数：

```json
{
  "train_data_path": "/path/to/train.csv",
  "prediction_length": 96,
  "context_length": 512,
  "finetune_mode": "lora",
  "learning_rate": 0.0001,
  "num_steps": 1000,
  "batch_size": 32,
  "selected_groups": [
    {
      "target": "value1",
      "covariates": ["value2", "value3"]
    },
    {
      "target": "value2",
      "covariates": ["value1", "value4"]
    }
  ]
}
```

发布模型时，应把这些和推理相关的信息写入发布目录下的 `metadata.json`。

### 3.2 推理接口中的 `cov_group` 改为可选

推理接口应支持两种模式：

#### 普通模式：使用模型自带配置

请求：

```json
{
  "model_path": "...",
  "csv_path": "..."
}
```

后端行为：

- 读取 `<model_path>/metadata.json`
- 使用其中的 `selected_groups`
- 使用其中的 `prediction_length`
- 使用其中的 `context_length`

#### 高级模式：允许显式覆盖

请求：

```json
{
  "model_path": "...",
  "csv_path": "...",
  "cov_group": [
    {
      "target": "value1",
      "covariates": ["value2"]
    }
  ],
  "prediction_length": 5,
  "context_length": 128
}
```

后端行为：

- 如果请求显式传入 `cov_group`，则优先使用请求中的 `cov_group`
- 如果请求显式传入 `prediction_length`，则优先使用请求中的值
- 如果请求显式传入 `context_length`，则优先使用请求中的值
- 未传入的字段从 `metadata.json` 中补齐

---

## 4. 建议新增 metadata.json 格式

发布后的模型目录建议如下：

```text
release/
└── models/
    └── user_10001/
        └── v1.0.0/
            └── train_job_20260121103000/
                ├── metadata.json
                ├── finetuned-ckpt_value1/
                └── finetuned-ckpt_value2/
```

建议 `metadata.json` 内容格式：

```json
{
  "job_id": "train_job_20260121103000",
  "user_id": "10001",
  "version": "1.0.0",
  "model_type": "chronos-2",
  "prediction_length": 96,
  "context_length": 512,
  "selected_groups": [
    {
      "target": "value1",
      "covariates": ["value2", "value3"],
      "model_dir": "finetuned-ckpt_value1"
    },
    {
      "target": "value2",
      "covariates": ["value1", "value4"],
      "model_dir": "finetuned-ckpt_value2"
    }
  ],
  "created_from_train_request": {
    "finetune_mode": "lora",
    "learning_rate": 0.0001,
    "num_steps": 1000,
    "batch_size": 32
  }
}
```

说明：

- `selected_groups` 是推理时最重要的信息。
- `model_dir` 用于明确每个 target 对应的子模型目录。
- `prediction_length` 和 `context_length` 可作为推理默认值。
- `created_from_train_request` 用于记录训练来源，便于调试和追溯。
- 不建议在 `metadata.json` 中保存训练数据路径等敏感或强环境相关路径，除非确有需要。

---

## 5. 需要修改的模块

具体文件名以当前项目实际代码为准，可优先检查以下模块：

```text
app/
├── api/
│   └── finetune.py
├── schemas/
│   └── request.py
├── schemas/
│   └── response.py
├── services/
│   ├── job_service.py
│   ├── model_service.py
│   └── dataset_service.py
└── tests/
```

可能涉及的功能点：

- 模型发布逻辑
- 推理请求 schema
- 推理服务逻辑
- 模型元数据读写
- CSV 列校验
- 接口测试

---

## 6. 具体 TODO

### 6.1 定义模型元数据读写工具

- [ ] 新增或复用一个模型元数据工具模块，例如：

```text
app/services/model_metadata_service.py
```

- [ ] 实现 `write_model_metadata(model_path, metadata)`。
- [ ] 实现 `load_model_metadata(model_path)`。
- [ ] 当 `<model_path>/metadata.json` 不存在时，返回清晰错误。
- [ ] 当 `metadata.json` 不是合法 JSON 时，返回清晰错误。
- [ ] 校验 metadata 中至少包含：
  - [ ] `selected_groups`
  - [ ] `prediction_length`
  - [ ] `context_length`

---

### 6.2 修改模型发布逻辑

在 `/api/model/publish` 或 `/v1/finetune/jobs/release` 对应的发布逻辑中：

- [ ] 发布模型目录时，除了复制模型文件，也要生成 `metadata.json`。
- [ ] 从训练任务的 `request_json` 中读取：
  - [ ] `selected_groups`
  - [ ] `prediction_length`
  - [ ] `context_length`
  - [ ] `finetune_mode`
  - [ ] `learning_rate`
  - [ ] `num_steps`
  - [ ] `batch_size`
- [ ] 从训练任务中读取：
  - [ ] `job_id`
  - [ ] `target_model_map`
  - [ ] `model_paths`
- [ ] 为每个 selected group 补充对应的 `model_dir`。
- [ ] 将最终 metadata 写入：

```text
<released_model_path>/metadata.json
```

- [ ] 如果发布目录已存在并被覆盖，确保新的 `metadata.json` 也被重新生成。
- [ ] 确保两个发布接口的行为一致：
  - [ ] `/v1/finetune/jobs/release`
  - [ ] `/api/model/publish`

---

### 6.3 修改推理请求 Schema

当前 `/api/model/infer` 请求中，以下字段可能是必需的：

- `cov_group`
- `prediction_length`
- `context_length`

需要修改为：

- [ ] `cov_group` 改为可选。
- [ ] `prediction_length` 改为可选。
- [ ] `context_length` 改为可选。
- [ ] `model_path` 仍然必需。
- [ ] `csv_path` 仍然必需。

推荐语义：

```python
cov_group = request.cov_group or metadata.selected_groups
prediction_length = request.prediction_length or metadata.prediction_length
context_length = request.context_length or metadata.context_length
```

注意：

- [ ] 如果请求中没有传 `cov_group`，但 metadata 中也没有 `selected_groups`，应返回错误。
- [ ] 如果请求中没有传 `prediction_length`，但 metadata 中也没有，应该返回错误。
- [ ] 如果请求中没有传 `context_length`，但 metadata 中也没有，应该返回错误。
- [ ] 如果请求中传入了无效的正整数，也应该返回错误。

---

### 6.4 修改推理服务逻辑

推理逻辑应改为：

1. 接收 `model_path` 和 `csv_path`。
2. 校验 `model_path` 是否存在。
3. 读取 `<model_path>/metadata.json`。
4. 合并请求参数和 metadata 参数。
5. 加载 CSV。
6. 根据最终的 `cov_group` 检查列是否存在。
7. 根据每个 target 找到对应子模型。
8. 逐组执行滚动窗口推理。
9. 返回结果。

需要完成：

- [ ] 推理入口先加载 metadata。
- [ ] 请求参数优先级高于 metadata。
- [ ] metadata 只提供默认配置。
- [ ] 保持现有返回格式不变：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "predictions": [
      {
        "target": "value1",
        "prediction": [0.1, 0.2, 0.3],
        "actual": [10.1, 10.4, 10.9]
      }
    ]
  }
}
```

- [ ] 保持现有错误格式不变：

```json
{
  "code": 400,
  "message": "history length is insufficient",
  "data": null
}
```

---

### 6.5 子模型目录查找规则

建议优先使用 metadata 中的 `model_dir` 字段。

查找优先级：

1. 如果 group 中有 `model_dir`：
   - 使用 `<model_path>/<model_dir>`
2. 如果没有 `model_dir`：
   - fallback 到旧规则：`<model_path>/finetuned-ckpt_<target>`

需要完成：

- [ ] 实现 `resolve_model_dir(model_path, group)`。
- [ ] 优先读取 `group["model_dir"]`。
- [ ] 缺失时兼容旧目录命名。
- [ ] 找不到子模型目录时返回清晰错误：

```json
{
  "code": 404,
  "message": "model for target 'value1' not found",
  "data": null
}
```

---

### 6.6 CSV 列校验

在正式推理前，应进行完整列校验。

需要完成：

- [ ] 检查每个 `target` 是否存在于 CSV 列中。
- [ ] 检查每个 `covariate` 是否存在于 CSV 列中。
- [ ] 检查 target 和 covariates 是否为数值列。
- [ ] 如果缺列，返回清晰错误，例如：

```json
{
  "code": 400,
  "message": "missing column which model required: value3",
  "data": null
}
```

- [ ] 如果列不是数值列，返回清晰错误，例如：

```json
{
  "code": 400,
  "message": "column 'value3' must be numeric",
  "data": null
}
```

---

### 6.7 新增模型信息接口

建议新增：

```http
GET /api/model/info?model_path=...
```

用途：

- 给前端展示模型能预测哪些变量。
- 给用户确认该模型训练时使用了哪些相关组。
- 避免用户盲目选择模型。

响应示例：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
    "targets": ["value1", "value2"],
    "selected_groups": [
      {
        "target": "value1",
        "covariates": ["value2", "value3"]
      },
      {
        "target": "value2",
        "covariates": ["value1", "value4"]
      }
    ],
    "prediction_length": 96,
    "context_length": 512
  }
}
```

需要完成：

- [ ] 新增 `/api/model/info` 路由。
- [ ] 读取 `metadata.json`。
- [ ] 返回 `targets`。
- [ ] 返回 `selected_groups`。
- [ ] 返回默认 `prediction_length`。
- [ ] 返回默认 `context_length`。
- [ ] 如果 `metadata.json` 不存在，返回清晰错误。

---

## 7. 向后兼容要求

本次改造必须保持向后兼容。

需要保证：

- [ ] 旧的推理请求仍然可用，即仍然允许前端传入完整 `cov_group`。
- [ ] 旧的 `prediction_length` / `context_length` 显式传参仍然可用。
- [ ] 如果发布目录没有 `metadata.json`，但请求中完整传入了 `cov_group`、`prediction_length`、`context_length`，可以继续使用旧逻辑推理。
- [ ] 如果发布目录没有 `metadata.json`，且请求中缺少必要参数，应返回明确错误，而不是程序崩溃。
- [ ] 现有返回结构不应破坏前端兼容性。

推荐判断逻辑：

```python
metadata = try_load_metadata(model_path)

cov_group = request.cov_group
if cov_group is None:
    cov_group = metadata.selected_groups if metadata else None

prediction_length = request.prediction_length
if prediction_length is None:
    prediction_length = metadata.prediction_length if metadata else None

context_length = request.context_length
if context_length is None:
    context_length = metadata.context_length if metadata else None

if cov_group is None:
    return error("cov_group is required when metadata.json is missing")

if prediction_length is None:
    return error("prediction_length is required when metadata.json is missing")

if context_length is None:
    return error("context_length is required when metadata.json is missing")
```

---

## 8. 建议测试用例

### 8.1 metadata 写入测试

- [ ] 发布模型后，检查发布目录下存在 `metadata.json`。
- [ ] 检查 metadata 中包含 `selected_groups`。
- [ ] 检查 metadata 中包含 `prediction_length`。
- [ ] 检查 metadata 中包含 `context_length`。
- [ ] 检查每个 group 中包含正确的 `target` 和 `covariates`。
- [ ] 检查每个 group 中包含正确的 `model_dir`。

---

### 8.2 简化推理请求测试

- [ ] 只传 `model_path` 和 `csv_path`，可以成功推理。
- [ ] 返回结果中包含所有 metadata 中记录的 targets。
- [ ] 返回结果格式与原接口一致。

---

### 8.3 覆盖参数测试

- [ ] 请求中显式传入 `cov_group` 时，使用请求中的 `cov_group`。
- [ ] 请求中显式传入 `prediction_length` 时，使用请求中的值。
- [ ] 请求中显式传入 `context_length` 时，使用请求中的值。
- [ ] 未显式传入的参数仍从 metadata 中读取。

---

### 8.4 旧模型兼容测试

- [ ] 模型目录中没有 `metadata.json`，但请求中完整传入 `cov_group`、`prediction_length`、`context_length` 时，仍可推理。
- [ ] 模型目录中没有 `metadata.json`，且请求中缺少 `cov_group` 时，返回明确错误。
- [ ] 模型目录中没有 `metadata.json`，且请求中缺少 `prediction_length` 时，返回明确错误。
- [ ] 模型目录中没有 `metadata.json`，且请求中缺少 `context_length` 时，返回明确错误。

---

### 8.5 CSV 校验测试

- [ ] CSV 缺少 target 列时，返回 400。
- [ ] CSV 缺少 covariate 列时，返回 400。
- [ ] CSV 对应列不是数值列时，返回 400。
- [ ] CSV 行数小于或等于 `context_length` 时，返回 `history length is insufficient`。

---

### 8.6 子模型目录测试

- [ ] metadata 中存在 `model_dir` 时，优先使用 `model_dir`。
- [ ] metadata 中没有 `model_dir` 时，fallback 到 `finetuned-ckpt_<target>`。
- [ ] 子模型目录不存在时，返回 404。

---

### 8.7 模型信息接口测试

- [ ] `/api/model/info?model_path=...` 可以返回模型元数据。
- [ ] 返回结果中包含 `targets`。
- [ ] 返回结果中包含 `selected_groups`。
- [ ] 返回结果中包含 `prediction_length`。
- [ ] 返回结果中包含 `context_length`。
- [ ] metadata 缺失时返回清晰错误。

---

## 9. 前端改造建议

前端可以改为：

- [ ] 推理页面只要求用户选择：
  - [ ] 已发布模型
  - [ ] 推理 CSV 文件
- [ ] 调用 `/api/model/info` 展示该模型信息：
  - [ ] 可预测变量
  - [ ] 每个变量依赖的 covariates
  - [ ] 默认 context_length
  - [ ] 默认 prediction_length
- [ ] 默认不展示 `cov_group` 编辑入口。
- [ ] 可以保留一个“高级设置”区域，允许用户覆盖：
  - [ ] `cov_group`
  - [ ] `prediction_length`
  - [ ] `context_length`

---

## 10. 验收标准

本次改造完成后，应满足：

- [ ] 用户不需要记住训练时的变量相关组。
- [ ] 用户只传 `model_path` 和 `csv_path` 即可推理。
- [ ] 模型发布目录下包含 `metadata.json`。
- [ ] 推理接口能自动读取 metadata。
- [ ] 前端可以通过 `/api/model/info` 获取模型可解释信息。
- [ ] 老接口调用方式仍然可用。
- [ ] 缺少 metadata 或 CSV 列异常时，后端返回清晰错误。
- [ ] 所有新增和原有测试通过。

---

## 11. 推荐实现顺序

建议按以下顺序实现：

1. [ ] 新增 metadata 读写工具。
2. [ ] 修改发布逻辑，生成 `metadata.json`。
3. [ ] 修改推理请求 schema，将 `cov_group`、`prediction_length`、`context_length` 改为可选。
4. [ ] 修改推理逻辑，从 metadata 中补齐缺失参数。
5. [ ] 增加 CSV 列校验和子模型目录解析逻辑。
6. [ ] 新增 `/api/model/info` 接口。
7. [ ] 补充测试。
8. [ ] 更新 README 中 `/api/model/infer` 和 `/api/model/info` 的说明。

---

## 12. README 更新建议

README 中 `/api/model/infer` 部分应改为：

### 简化推理请求

```json
{
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "csv_path": "/abs/path/to/new_data.csv"
}
```

说明：

- 如果 `model_path` 下存在 `metadata.json`，后端会自动读取训练时保存的 `selected_groups`、`prediction_length` 和 `context_length`。
- 用户不再需要手动输入 `cov_group`。

### 高级推理请求

```json
{
  "model_path": "/abs/path/to/release/models/user_10001/v1.0.0/train_job_20260121103000",
  "csv_path": "/abs/path/to/new_data.csv",
  "cov_group": [
    {
      "target": "value1",
      "covariates": ["value2"]
    }
  ],
  "prediction_length": 5,
  "context_length": 128
}
```

说明：

- 如果传入 `cov_group`，则覆盖 metadata 中的 `selected_groups`。
- 如果传入 `prediction_length` 或 `context_length`，则覆盖 metadata 中的默认值。

---

## 13. 注意事项

- 不要删除旧接口能力，只把原来的必填参数改为可选参数。
- metadata 是发布模型的一部分，应随着模型目录一起复制、备份和迁移。
- 不建议让前端自己拼接 `finetuned-ckpt_<target>` 路径。
- 不建议让用户自己记忆训练时的相关组。
- 训练时的 `selected_groups` 是模型语义的一部分，应该由后端持久化。
- 如果后续支持模型注册表，可以把 metadata 同步写入数据库，但当前阶段先写入发布目录即可。
