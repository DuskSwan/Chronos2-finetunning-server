"""模拟训练服务。

提供假的训练函数用于测试和调试训练工作流。
设计上便于后续替换为真实的 Chronos-2 或其他训练器。
"""

import random
import time
from pathlib import Path
from typing import Any, Dict


def mock_train(
    config: Dict[str, Any],
    steps: int = 5,
) -> str:
    """模拟微调过程。

    此函数模拟一个假的微调训练，不实际对数据进行处理或模型训练。

    Args:
        config: 训练配置字典，包含 output_dir 等信息。
        steps: 训练步数。默认 5。

    Returns:
        模型检查点的路径（假的）。

    例：
        model_path = mock_train({"output_dir": "/path/to/artifacts/job-id"})
    """
    output_dir = Path(config.get("output_dir", "./artifacts"))
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模拟训练过程
    print(f"[模拟训练器] 开始训练，总步数: {steps}")
    
    for step in range(1, steps + 1):
        # 模拟每步处理时间 0.2 ~ 0.5 秒
        sleep_time = random.uniform(0.2, 0.5)
        time.sleep(sleep_time)
        
        # 模拟损失下降
        loss = 1.0 - (step / steps) * 0.7  # 从 1.0 下降到 ~0.3
        loss = loss + random.uniform(-0.02, 0.02)  # 加入一点随机波动
        loss = max(loss, 0.1)  # 防止负值
        
        print(f"[模拟训练器] 第 {step}/{steps} 步，损失: {loss:.4f}")
    
    # 返回假的模型路径
    model_path = str(output_dir / "finetuned-ckpt")
    print(f"[模拟训练器] 训练完成，模型路径: {model_path}")
    
    return model_path
