# ExternalNode 集成型二进制推理改造 TODO（确认版）

## 0. 已确认约束（本项目最终口径）

以下约束已由需求方确认，后续实现不再反复讨论：

1. `--model-path` 在本项目中是必填参数（尽管通用规范写可选）。
2. 运行时有效业务输入只有：模型路径 + 上游发送的数据表（`list[dict]`）。
3. `prediction_length/context_length/cov_group` 不再由请求传入，全部依赖 `metadata.json`。
4. 返回 `data` 结构固定为：`{target: prediction}`。
   - 不返回 `actual`
   - 不返回 `metadata`
   - 不返回 `request_id`
5. 输入列处理规则：
   - 多余列忽略
   - 仅提取模型需要列并进行数值处理
   - 缺失必需列直接报错
6. 节点模式下 `metadata.json` 必须存在，缺失即失败。
7. 仅支持 REQ 一问一答模式；不考虑并发。
8. 进程生命周期由外部工具管理；本程序不承担会话关闭协商。

---

## 1. 背景与目标

当前目标已变更：

- 不再交付“边端独立 CLI 批处理程序”；
- 改为交付“可被 ExternalNode 拉起并通过 ZMQ 通信的推理进程”。

该二进制程序启动后需要：

1. 解析参数 `--model-path --zmq-endpoint --zmq-protocol`
2. 在指定 endpoint 绑定 ZMQ 服务端 socket
3. 接收上游每次发送的一条 JSON 字符串（内容为“字典列表”，等价多行 CSV）
4. 执行一次推理（每个 target 产出预测）
5. 按约定 JSON 结构返回结果

首版范围：

- 只支持 `REQ` 协议（服务端 `REP`）
- 明确不支持 `DEALER`（服务端 `ROUTER`）

---

## 2. 外部协议约束（来自二进制节点规范）

### 2.1 启动参数

平台会传入：

- `--model-path <abs_path>`
- `--zmq-endpoint <endpoint>`
- `--zmq-protocol <REQ|DEALER>`

本项目要求：

- `--model-path` 必填
- `--zmq-endpoint` 必填
- `--zmq-protocol` 必填
- 不应自行发明替代参数名

### 2.2 输入消息（单条）

- 类型：JSON 字符串
- 内容：`list[dict]`，每个 dict 表示一行记录
- 示例：`[{"timestamp":"...","value":12.3}, {...}]`

### 2.3 输出消息（REQ 模式）

每条输入对应一条输出。

成功响应格式：

```json
{
  "code": 200,
  "type": "timeseries",
  "version": "1.0",
  "data": {
    "value1": [0.1, 0.2],
    "value2": [0.3, 0.4]
  },
  "message": "success"
}
```

失败响应格式：

```json
{
  "code": 400,
  "type": "timeseries",
  "version": "1.0",
  "data": {},
  "message": "...error..."
}
```

---

## 3. 与当前实现的差异

当前已有：

- `run_inference(...)`：基于 `model_path + csv_path` 推理
- CLI：基于文件输入输出（`--csv-path --output-path`）

本次需要新增：

- ZMQ 服务循环
- 将 `list[dict]` 输入转换为 DataFrame（不依赖 CSV 文件）
- 以 ExternalNode 约定 JSON 直接回包，而不是落盘文件
- 强制 metadata 存在并参与配置补齐

因此需要抽象：

- 新增“DataFrame 直接推理”的公共入口，供 ZMQ 服务复用

---

## 4. 总体设计

### 4.1 模块结构建议

```text
app/
├── services/
│   ├── inference_service.py          # 现有核心推理
│   └── inference_runtime_service.py  # 新增：DataFrame 入参推理适配
├── runtime/
│   └── zmq_infer_server.py           # 新增：ZMQ 服务主循环
└── cli/
    └── infer_node_cli.py             # 新增：二进制入口
```

### 4.2 调用链

```text
ExternalNode 启动进程
  -> infer_node_cli 解析参数
  -> 初始化 ZMQ (REQ -> REP)
  -> 收包(JSON list[dict])
  -> 反序列化 + DataFrame 校验
  -> 调用公共推理服务
  -> 回包(JSON)
```

---

## 5. 参数与运行规则

### 5.1 必需参数

- `--model-path`
- `--zmq-endpoint`
- `--zmq-protocol`

### 5.2 协议限制

- `--zmq-protocol=REQ`：正常启动（绑定 `REP`）
- `--zmq-protocol=DEALER`：直接失败退出，并提示“当前版本不支持 DEALER”

### 5.3 metadata 规则

- `<model_path>/metadata.json` 必须存在
- 不允许通过请求额外传入 `prediction_length/context_length/cov_group`
- 推理所需配置全部从 metadata 获取

---

## 6. 消息处理规则

### 6.1 入站校验

- JSON 反序列化失败 -> `code=400`
- 顶层不是 list -> `code=400`
- list 元素不是 dict -> `code=400`
- 空 list -> `code=400`

### 6.2 数据映射

- `list[dict]` -> `pandas.DataFrame`
- 输入可能包含额外列：允许并忽略
- 仅按模型需求列参与推理
- 缺少必需列：直接失败并回包错误
- 必需列非数值：直接失败并回包错误

### 6.3 出站格式

成功：

- `code=200`
- `data` 固定为 `{target: prediction_list}`

失败：

- `code` 依据错误类型返回（建议 400/404/500）
- `data={}`
- `message` 提供可读原因

---

## 7. 实施清单

### 7.1 推理服务改造（P0）

- [ ] 在 `inference_service` 基础上新增 DataFrame 入参推理能力
- [ ] 提供统一内部函数，避免 CSV 与 DataFrame 两套逻辑分叉
- [ ] 强制 metadata 必须存在（节点模式）
- [ ] 输出结构适配 `{target: prediction}`
- [ ] 保持现有 `/api/model/infer` 与已有测试兼容

### 7.2 ZMQ 运行时（P0）

- [ ] 新增 `runtime/zmq_infer_server.py`
- [ ] 实现 REQ 模式服务端（REP bind）
- [ ] 实现单请求单响应循环
- [ ] 实现优雅错误回包（不因单条坏消息崩进程）

### 7.3 节点入口（P0）

- [ ] 新增 `cli/infer_node_cli.py`
- [ ] 解析 `--model-path --zmq-endpoint --zmq-protocol`
- [ ] `DEALER` 直接拒绝并退出非 0
- [ ] 启动 ZMQ 服务主循环

### 7.4 打包脚本（P1）

- [ ] 新增/调整打包脚本，入口切到 `app/cli/infer_node_cli.py`
- [ ] Windows onedir：`chronos_infer_node.exe`
- [ ] Linux onedir：`chronos_infer_node`

### 7.5 文档（P1）

- [ ] README 增加“ExternalNode 集成模式”章节
- [ ] 给出 REQ 启动示例
- [ ] 给出请求/响应 JSON 示例
- [ ] 明确声明：当前不支持 DEALER
- [ ] 明确声明：metadata 必须存在

### 7.6 测试（P1）

- [ ] 新增 `tests/test_zmq_infer_node.py`
- [ ] 覆盖 REQ 正常收发
- [ ] 覆盖非法 JSON
- [ ] 覆盖空 list
- [ ] 覆盖列缺失/类型错误
- [ ] 覆盖 metadata 缺失
- [ ] 覆盖 `--zmq-protocol=DEALER` 启动失败

---

## 8. 错误码与兼容建议

建议回包 code：

- `200`：成功
- `400`：输入数据错误/参数错误
- `404`：模型路径或子模型不存在
- `500`：内部推理错误

建议 message：

- 面向节点集成日志可读
- 不默认暴露完整 Python traceback

---

## 9. 验收标准

- [ ] 可执行程序可被 ExternalNode 参数拉起
- [ ] `REQ` 模式下可稳定一问一答
- [ ] 输入 `list[dict]` 可完成推理并返回 `{target: prediction}`
- [ ] 错误请求不会导致进程退出
- [ ] metadata 缺失时明确报错
- [ ] `DEALER` 明确返回“不支持”并退出非 0
- [ ] 保持现有 API 推理能力与测试通过

---

## 10. 推荐执行顺序

1. [ ] 抽象 DataFrame 推理入口（先不动 ZMQ）
2. [ ] 实现 metadata 强制存在的节点模式校验
3. [ ] 实现 ZMQ REQ/REP 服务循环
4. [ ] 新增节点 CLI 入口并联调
5. [ ] 增加自动化测试
6. [ ] 更新打包脚本（Win/Linux）与 README
7. [ ] 做一次本机 REQ 集成冒烟验证

---

## 11. 非目标（当前阶段不做）

- [ ] DEALER/ROUTER 多帧会话
- [ ] 一请求多响应流式输出
- [ ] GPU 设备切换参数化
- [ ] 动态热加载模型
- [ ] 复杂鉴权/加密通信
- [ ] 进程关闭协商（交由外部工具直接终止）

---

## 12. 结论

新方案的核心是“从文件型 CLI”转向“长驻 ZMQ 节点进程”。
首版先把 REQ 一问一答模式做稳定，确保可与 ExternalNode 完成端到端集成；DEALER 明确暂不支持并给出可读错误即可。
