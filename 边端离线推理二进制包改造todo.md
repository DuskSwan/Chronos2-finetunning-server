# 边端离线推理二进制包改造 TODO（优化版）

## 1. 现状结论（基于当前代码）

当前项目已经具备离线 CLI 改造的关键基础：

- 已有公共推理服务：`app/services/inference_service.py::run_inference(...)`
- 该服务已不依赖 FastAPI Request/Response，可被 API 与 CLI 共同复用
- 已支持从 `<model_path>/metadata.json` 自动补齐：
  - `selected_groups`
  - `prediction_length`
  - `context_length`
- 已支持子模型目录解析：优先 `model_dir`，回退 `finetuned-ckpt_<target>`
- 已有较完整的推理接口兼容测试：`tests/test_inference_api.py`

因此，本次重点不应再放在“抽离推理逻辑”，而应聚焦：

1. 新增 CLI 入口与参数规范
2. 输出文件与退出码规范
3. 打包可运行性（PyInstaller onedir）
4. 离线交付文档与验收

---

## 2. 目标重述（收敛版）

目标：提供一个可离线部署的推理可执行程序（优先 `chronos_infer`），通过命令行输入模型与数据路径完成推理。

必须满足：

- 不启动 FastAPI 服务也可运行
- 不内置模型权重和业务数据
- 默认复用 metadata 自动配置
- 成功输出结构化 JSON 文件
- 用退出码表达成功/失败
- 可被 PyInstaller `onedir` 方式打包

---

## 3. 与现有 TODO 的差异与优化点

### 3.1 删除/降级的项

- “抽离公共推理服务”不再作为主任务（已完成）
- “保证 `/api/model/infer` 不被破坏”改为回归项，不作为主实现项
- “CLI 输出尽量与 HTTP 完全一致”改为“结构相近 + CLI 语义清晰”
  - 建议保留 `code/message/data`，便于统一消费

### 3.2 需要新增的关键项

- 明确 CLI 参数与 `run_inference` 参数映射
- 增加 `--targets` 的语义定义（从 metadata/cov_group 里筛选，不是新增推理组）
- 明确路径创建策略：`--output-path` 父目录自动创建
- 明确错误边界：业务错误（4xx语义）vs 内部错误（5xx语义）
- 增加“最小打包验证脚本”和“干净环境冒烟验证步骤”

---

## 4. CLI 设计（按当前项目可直接实现）

### 4.1 命令示例

```bash
chronos_infer \
  --model-path ./models/train_job_001 \
  --csv-path ./data/input.csv \
  --output-path ./result/output.json
```

### 4.2 参数定义

必填参数：

- `--model-path`：发布模型目录
- `--csv-path`：输入 CSV 路径
- `--output-path`：输出 JSON 文件路径

可选参数：

- `--prediction-length`：覆盖 metadata 默认值
- `--context-length`：覆盖 metadata 默认值
- `--targets`：逗号分隔，仅推理指定 target（如 `value1,value2`）
- `--verbose`：输出调试日志（含异常堆栈）
- `--version`：输出版本

暂不建议首版支持：

- `--device`（当前 `run_inference` 内部固定 `cpu`）
- `--format csv`（先统一 JSON，后续扩展）

### 4.3 输出与退出码

输出 JSON 建议：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "...",
    "csv_path": "...",
    "predictions": []
  }
}
```

退出码：

- `0`：成功
- `2`：参数错误
- `3`：文件/路径错误
- `4`：推理业务错误（如列缺失、历史长度不足）
- `1`：其他未预期错误

---

## 5. 实施清单（更新后）

### 5.1 新增 CLI 入口（P0）

- [ ] 新增文件：`app/cli/infer_cli.py`
- [ ] 使用 `argparse` 解析参数
- [ ] 调用 `run_inference(...)`
- [ ] 将结果写入 `--output-path`
- [ ] 返回规范退出码

### 5.2 适配 metadata 与 targets 过滤（P0）

- [ ] 当未显式传 `cov_group` 时，复用 `run_inference` 现有 metadata 补齐能力
- [ ] 实现 `--targets` 过滤：
  - [ ] 若 metadata/请求组中不含指定 target，报错
  - [ ] 保持原有组内 covariates 不变

### 5.3 错误与日志（P0）

- [ ] 捕获 `InferenceError`，输出用户可读错误
- [ ] `--verbose` 下打印 traceback
- [ ] 默认模式不打印长堆栈

### 5.4 打包支持（P1）

- [ ] 新增 `scripts/build_infer_exe.ps1`（PyInstaller onedir）
- [ ] 在 README 给出打包命令、产物目录、运行示例
- [ ] 验证在未启动 API 服务时可执行推理

### 5.5 测试（P1）

- [ ] 新增 `tests/test_infer_cli.py`
- [ ] 覆盖最小成功路径
- [ ] 覆盖 metadata 缺失+参数不足失败
- [ ] 覆盖 `--targets` 过滤
- [ ] 覆盖输出文件生成与退出码

---

## 6. 建议目录结构

```text
app/
├── cli/
│   └── infer_cli.py
├── services/
│   └── inference_service.py
scripts/
└── build_infer_exe.ps1
tests/
└── test_infer_cli.py
```

---

## 7. README 必加内容

- CLI 用途与适用场景（离线、边端、批处理）
- 最小运行命令
- 参数说明与优先级（显式参数 > metadata）
- 输出 JSON 示例
- 常见错误与排查
- 打包步骤（PyInstaller onedir）
- 离线交付目录示例

---

## 8. 风险与规避

- 依赖体积大（Torch/Transformers）
  - 规避：首版仅 `onedir`，不追求单文件
- 不同机器 CUDA/驱动差异
  - 规避：首版默认 CPU 推理
- 模型目录结构不一致
  - 规避：继续沿用 `model_dir` 优先 + 旧命名回退

---

## 9. 验收标准（可执行）

- [ ] `python -m app.cli.infer_cli --model-path ... --csv-path ... --output-path ...` 可运行成功
- [ ] 不启动 FastAPI 也可推理
- [ ] `dist/chronos_infer/` 可在目标机离线运行（提供模型与 CSV）
- [ ] 输出 JSON 包含 `code/message/data.predictions`
- [ ] 失败时退出码非 `0` 且错误可读
- [ ] 原 `tests/test_inference_api.py` 保持通过

---

## 10. 推荐执行顺序（按性价比）

1. [ ] 实现 CLI 入口（先 Python 直跑）
2. [ ] 补齐 CLI 测试（成功/失败/targets）
3. [ ] 增加打包脚本并本机验证 `onedir`
4. [ ] 更新 README 与交付示例目录
5. [ ] 做一次“干净环境”冒烟验证

---

## 11. 结论

你原始 todo 的总体方向是正确的；主要问题是“把已完成项当成主改造项”。优化后应把重心转移到 CLI、退出码、打包验证和离线交付文档，这样能最快落地边端可用版本。
