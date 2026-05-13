# 建议在项目根目录执行；Windows 下用 curl.exe 避免 PowerShell 的 curl 别名问题
# 你的项目路径示例：d:\GitRepo\ts_model_train_and_finetune

# 0) 可选：如果你启用了 API_BEARER_TOKEN，就加这个头；没启用就删掉 -H 这一行
# -H "Authorization: Bearer spec-token"

# 1) 创建训练任务（使用本地 mock_train_data.csv）
curl.exe -X POST "http://127.0.0.1:8000/v1/finetune/jobs" ^
  -H "Content-Type: application/json" ^
  -d "{\"train_data_path\":\"d:\\GitRepo\\ts_model_train_and_finetune\\mock_train_data.csv\",\"prediction_length\":2,\"context_length\":32,\"selected_groups\":[{\"target\":\"value1\",\"covariates\":[\"value2\",\"value3\"]}]}"

# 响应会有 job_id，比如：{"job_id":"...","status":"queued"}

# 2) 查训练状态（把 <JOB_ID> 替换成上一步返回值）
curl.exe "http://127.0.0.1:8000/v1/finetune/jobs/<JOB_ID>"

# 3) 训练完成后发布模型（把 <JOB_ID> 替换）
curl.exe -X POST "http://127.0.0.1:8000/api/model/publish" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":10001,\"version\":\"1.0.0\",\"job_id\":\"<JOB_ID>\"}"

# 响应会有 model_path，比如 data.model_path

# 4) 常规推理（把 <MODEL_PATH> 替换成发布返回的绝对路径）
curl.exe -X POST "http://127.0.0.1:8000/api/model/infer" ^
  -H "Content-Type: application/json" ^
  -d "{\"model_path\":\"<MODEL_PATH>\",\"csv_path\":\"d:\\GitRepo\\ts_model_train_and_finetune\\mock_train_data.csv\"}"

# 5) 分段推理配置查询（新接口）
curl.exe "http://127.0.0.1:8000/api/model/infer/config?model_path=<MODEL_PATH>"

# 6) 分段推理（新接口，示例首段）
curl.exe -X POST "http://127.0.0.1:8000/api/model/infer/chunk" ^
  -H "Content-Type: application/json" ^
  -d "{\"task_id\":\"task_demo_001\",\"model_path\":\"<MODEL_PATH>\",\"is_last_segment\":false,\"segment\":[{\"time\":47,\"value1\":0.012157337,\"value2\":0.017334247,\"value3\":0.011365411,\"value4\":0.02653957},{\"time\":48,\"value1\":0.057906676,\"value2\":0.013979295,\"value3\":0.033833418,\"value4\":0.030267294},{\"time\":49,\"value1\":0.061,\"value2\":0.014,\"value3\":0.034,\"value4\":0.031},{\"time\":50,\"value1\":0.062,\"value2\":0.015,\"value3\":0.035,\"value4\":0.032}]}"

# 7) 分段推理末段（触发释放缓存）
curl.exe -X POST "http://127.0.0.1:8000/api/model/infer/chunk" ^
  -H "Content-Type: application/json" ^
  -d "{\"task_id\":\"task_demo_001\",\"model_path\":\"<MODEL_PATH>\",\"is_last_segment\":true,\"segment\":[{\"time\":51,\"value1\":0.063,\"value2\":0.016,\"value3\":0.036,\"value4\":0.033},{\"time\":52,\"value1\":0.064,\"value2\":0.017,\"value3\":0.037,\"value4\":0.034},{\"time\":53,\"value1\":0.065,\"value2\":0.018,\"value3\":0.038,\"value4\":0.035},{\"time\":54,\"value1\":0.066,\"value2\":0.019,\"value3\":0.039,\"value4\":0.036}]}"
