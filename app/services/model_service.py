
from pathlib import Path
from loguru import logger

from chronos import Chronos2Pipeline
from app.core.config import Settings

settings = Settings()
CHRONOS_PATH = Path(settings.raw_model_cache_dir).resolve()

def load_local_model(model_path: Path|str = CHRONOS_PATH, device: str = "cpu") -> Chronos2Pipeline:
    """
    从本地路径加载 Chronos-2 模型。
    """
    # 加载模型
    model_source = Path(model_path)
    logger.info("加载 Chronos-2 本地模型: path={} (cwd={})".format(
        str(model_source.resolve()),
        str(Path.cwd()),
    ))
    assert model_source.is_dir(), f'试图从 {model_source} 加载模型，但该路径不是目录'
    pipeline = Chronos2Pipeline.from_pretrained(
        model_source,
        local_files_only=True,
        device_map=device,
    )
    return pipeline

def download_model():
    pipeline = Chronos2Pipeline.from_pretrained(
        'amazon/chronos-2',
        )
    path = CHRONOS_PATH
    path.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(path)

def test_load_model():
    model = load_local_model()
    print(model)

if __name__ == '__main__':
    download_model()
    test_load_model()
