# 边端离线推理 CLI（含二进制包说明）

本文档覆盖：

- 离线推理 CLI：`app.cli.infer_cli`
- ExternalNode 集成模式（ZMQ REQ）：`app.cli.infer_node_cli`
- Windows/Linux 二进制打包与交付建议

## 离线推理 CLI

### 1. Python 方式运行

```bash
python -m app.cli.infer_cli \
  --model-path ./release/models/user_10001/v1.0.0/train_job_xxx \
  --csv-path ./data/input.csv \
  --output-path ./result/output.json
```

### 2. 参数说明

- `--model-path`：发布模型目录（必填）
- `--csv-path`：输入 CSV 路径（必填）
- `--output-path`：输出 JSON 路径（必填）
- `--prediction-length`：覆盖 `metadata.json` 默认值（可选）
- `--context-length`：覆盖 `metadata.json` 默认值（可选）
- `--targets`：仅推理指定 target，逗号分隔（可选）
- `--verbose`：输出详细错误堆栈（可选）
- `--version`：查看 CLI 版本

参数优先级：显式参数 > `metadata.json` 默认值。

### 3. 输出格式

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_path": "...",
    "csv_path": "...",
    "predictions": [
      {
        "target": "value1",
        "prediction": [0.1, 0.2],
        "actual": [10.1, 10.4]
      }
    ]
  }
}
```

### 4. 退出码

- `0`：成功
- `2`：参数错误
- `3`：路径/文件不存在
- `4`：推理业务错误
- `1`：未预期错误

## 二进制打包（onedir）

注意：二进制不可跨平台通用。Windows 产物需在 Windows 打包；Linux 产物需在 Linux 打包。

### Windows

```powershell
.\scripts\build_infer_exe.ps1
```

可选参数：

```powershell
.\scripts\build_infer_exe.ps1 -Clean
.\scripts\build_infer_exe.ps1 -PythonExe ".\.venv\Scripts\python.exe" -DistDir "dist"
```

产物：

```text
dist/
└── chronos_infer/
    ├── chronos_infer.exe
    └── ...runtime files...
```

运行示例：

```powershell
.\dist\chronos_infer\chronos_infer.exe --model-path .\models\job_x --csv-path .\data\input.csv --output-path .\result\output.json
```

### Linux

```bash
chmod +x ./scripts/build_infer_exe.sh
./scripts/build_infer_exe.sh
```

可选参数：

```bash
CLEAN=1 ./scripts/build_infer_exe.sh
PYTHON_EXE=./.venv/bin/python DIST_DIR=dist ./scripts/build_infer_exe.sh
```

产物：

```text
dist/
└── chronos_infer/
    ├── chronos_infer
    └── ...runtime files...
```

运行示例：

```bash
./dist/chronos_infer/chronos_infer --model-path ./models/job_x --csv-path ./data/input.csv --output-path ./result/output.json
```

## ExternalNode 集成模式（ZMQ REQ）

### 1. 启动方式

```bash
python -m app.cli.infer_node_cli \
  --model-path /abs/path/to/model_release_dir \
  --zmq-endpoint tcp://127.0.0.1:52345 \
  --zmq-protocol REQ
```

参数说明：

- `--model-path`：必填，模型目录（需包含 `metadata.json`）
- `--zmq-endpoint`：必填，ExternalNode 分配端点
- `--zmq-protocol`：必填，当前仅支持 `REQ`（`DEALER` 不支持）

### 2. 通信模式

- ExternalNode `REQ` <-> 本程序 `REP`
- 一问一答，每条输入返回一条 JSON

### 3. 输入格式

输入必须是 `list[dict]` JSON 字符串，例如：

```json
[
  {"timestamp": "2026-05-09 10:00:00", "value1": 1.1, "value2": 2.2},
  {"timestamp": "2026-05-09 10:00:01", "value1": 1.2, "value2": 2.3}
]
```

列规则：

- 仅使用模型所需列
- 多余列忽略
- 缺列报错

### 4. 输出格式

成功：

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

失败：

```json
{
  "code": 400,
  "type": "timeseries",
  "version": "1.0",
  "data": {},
  "message": "missing column which model required: value2"
}
```

### 5. 节点版二进制打包

Windows：

```powershell
.\scripts\build_infer_node.ps1
```

Linux：

```bash
chmod +x ./scripts/build_infer_node.sh
./scripts/build_infer_node.sh
```

产物：

```text
dist/
└── chronos_infer_node/
    ├── chronos_infer_node.exe
    ├── chronos_infer_node
    └── ...runtime files...
```

## 离线交付建议目录

```text
edge_package/
├── chronos_infer/
│   ├── chronos_infer.exe (Windows)
│   ├── chronos_infer (Linux)
│   └── ...
├── models/
│   └── train_job_xxx/
│       ├── metadata.json
│       └── finetuned-ckpt_<target>/
├── data/
│   └── input.csv
└── result/
```

说明：CLI 不内置模型与数据，均通过命令行路径传入。
