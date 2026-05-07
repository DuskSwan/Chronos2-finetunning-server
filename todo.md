# 模型推理接口改造 TODO（实施版）

## 1. 约束与边界
- [x] 现有训练、发布、查询接口路径、请求参数、响应主结构不变。
- [x] 新增推理接口单独成路由，不破坏旧接口语义。
- [x] `/api/*` 兼容接口遵循 `code/message/data` 与 `snake_case`。
- [x] 推理基于已发布模型目录，不依赖训练中的临时产物目录。
- [x] 发布接口返回发布目录绝对路径，便于推理接口直接定位模型。
- [x] 推理输入使用训练外新数据，并做严格参数校验。
- [ ] 首版外的扩展能力（批量推理）暂不实现，后续迭代。

## 2. 接口契约设计
- [x] 新增推理接口：`POST /api/model/infer`。
- [x] 推理请求参数定稿：`model_path`、`cov_group`、`prediction_length`、`csv_path`。
- [x] 推理响应主结构定稿：`code/message/data`。
- [x] `data.predictions` 结构定稿：`[{target, prediction}]`，按 `cov_group` 顺序返回。
- [x] `/api/model/publish` 返回绝对路径（替代原相对路径约定）。

## 3. 推理服务层改造
- [x] 新建 `app/services/inference_service.py`。
- [x] 复用模型加载能力（基于 `load_local_model`）。
- [x] 从 `csv_path` 读取推理数据并做列级校验。
- [x] 实现推理主流程：加载模型 -> 组织输入 -> 调用预测 -> 输出格式化。
- [x] 每个 `cov_group` 按 `target` 选择对应子模型目录 `finetuned-ckpt_<target>`。
- [x] 多目标推理按组逐个调用并聚合返回。

## 4. API 与鉴权
- [x] 新增 `app/api/inference.py` 路由。
- [x] 在 `app/main.py` 注册推理路由。
- [x] 推理接口接入 `Bearer Token` 鉴权。
- [x] 推理异常细分并统一错误码（模型不存在/CSV非法/输入缺失/推理失败）。

## 5. Schema 与文档
- [x] 在 `app/schemas/request.py` 增加推理请求 Schema 与校验。
- [x] 在 `app/schemas/response.py` 增加推理响应 Schema。
- [x] 更新 `README.md` 与接口文档（最后阶段统一更新）。

## 6. 测试计划
- [x] 接口成功用例：多 target 推理成功（使用 `mock_train_data.csv`）。
- [x] 接口失败用例：目标模型不存在。
- [x] 发布接口测试同步到绝对路径断言。
- [x] 接口失败用例补全：`model_path` 不存在、`csv_path` 不存在、历史长度不足、协变量不匹配。
- [x] 服务层单测补全：模型路径解析、输入校验、结果格式化。

## 7. 交付验收清单
- [x] 新推理接口可调用且返回结构稳定。
- [x] 旧训练/发布/查询接口行为保持兼容。
- [x] 服务层已独立封装推理逻辑。
- [x] 发布目录可作为推理模型定位依据。
- [x] README 与对外接口文档补齐后再做最终验收。
