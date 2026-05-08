"""业务逻辑服务层。

封装与任务相关的核心业务操作，如任务创建、状态更新等。
解耦数据库操作与 API 路由。
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
from datetime import datetime, timezone
from collections import defaultdict

from fastapi import HTTPException, status

from app.db.crud import (
    create_job,
    get_job_by_id,
    update_job_status,
    update_job_progress,
    mark_job_completed,
    mark_job_failed,
    list_recent_jobs,
    list_recent_jobs_by_status,
    set_cancel_requested,
    mark_job_cancelled,
    list_job_loss_points,
    list_jobs_by_status,
    delete_job_by_id,
    list_jobs,
)
from app.core.enums import JobStatus
from app.core.config import Settings
from app.schemas.response import (
    JobDetailResponse,
    JobProgressResponse,
    JobListItemResponse,
    JobListResponse,
    CancelJobResponse,
    DeleteJobResponse,
    BatchDeleteJobsResponse,
)
from app.services.queue_service import get_job_queue


def _deserialize_model_paths(value: Optional[str]) -> Optional[list[str]]:
    if value is None or not str(value).strip():
        return None
    parsed = json.loads(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return None


def _deserialize_target_model_map(value: Optional[str]) -> Optional[dict[str, str]]:
    if value is None or not str(value).strip():
        return None
    parsed = json.loads(value)
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    return None


def _extract_group_targets(request_json: Optional[str]) -> list[str]:
    """兼容逻辑：从请求 JSON 中提取 selected_groups 的 target 名称。"""
    if not request_json:
        return []
    try:
        payload = json.loads(request_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    groups = payload.get("selected_groups")
    if not isinstance(groups, list):
        return []

    targets: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        target = group.get("target")
        if isinstance(target, str) and target.strip():
            targets.append(target.strip())
    return targets


def _build_loss_metrics(db: Any, job_id: str, request_json: Optional[str]) -> dict[str, list[float]]:
    """构建任务 loss 曲线，按 target 名称返回。"""
    loss_points = list_job_loss_points(db, job_id)
    if not loss_points:
        return {}

    grouped: dict[str, list[Any]] = defaultdict(list)
    target_names = _extract_group_targets(request_json)
    for point in loss_points:
        target = str(getattr(point, "target", "") or "").strip()
        group_index = int(getattr(point, "group_index", 0))
        # 历史数据兜底：target 为空或仍是 group_n 时，优先还原为请求中的真实 target
        if (not target) or target.startswith("group_"):
            target = (
                target_names[group_index]
                if group_index < len(target_names)
                else f"group_{group_index + 1}"
            )
        grouped[target].append(point)

    metrics: dict[str, list[float]] = {}
    for target in sorted(grouped.keys()):
        metrics[target] = [float(point.loss) for point in grouped[target]]

    return metrics


def _build_target_model_map_fallback(
    request_json: Optional[str],
    model_paths: Optional[list[str]],
) -> Optional[dict[str, str]]:
    if not model_paths:
        return None
    targets = _extract_group_targets(request_json)
    if not targets:
        return {f"group_{idx + 1}": str(path) for idx, path in enumerate(model_paths)}
    mapping: dict[str, str] = {}
    for idx, path in enumerate(model_paths):
        key = targets[idx] if idx < len(targets) else f"group_{idx + 1}"
        mapping[key] = str(path)
    return mapping


def create_finetune_job(
    request_data: Dict[str, Any],
    settings: Settings,
) -> str:
    """创建微调任务并保存请求信息。

    Args:
        request_data: 创建任务的请求数据。
        settings: 应用配置对象。

    Returns:
        生成的 job_id。
    """
    import uuid
    from app.db.session import SessionLocal
    
    # 生成 job_id
    job_id = str(uuid.uuid4())
    
    # 确定输出目录
    output_root = Path(request_data.get("output_root") or settings.artifacts_root)
    output_dir = output_root / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取日志路径
    logs_dir = Path(settings.logs_root) / job_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(logs_dir / "train.log")
    
    # 保存请求 JSON（可配置关闭）
    if settings.save_request_artifacts:
        request_json_path = output_dir / "request.json"
        with open(request_json_path, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
    
    # 创建数据库记录
    db = SessionLocal()
    job = create_job(
        db=db,
        job_id=job_id,
        status=JobStatus.queued.value,
        request_json=json.dumps(request_data, ensure_ascii=False),
        output_dir=str(output_dir),
        log_path=log_path,
    )
    db.close()
    
    return job_id


def start_job_training(db: Any, job_id: str) -> None:
    """标记任务为运行状态。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
    """
    update_job_status(
        db=db,
        job_id=job_id,
        status=JobStatus.running.value,
        started_at=datetime.now(timezone.utc),
    )


def update_job_step(
    db: Any,
    job_id: str,
    current_step: int,
    max_steps: int,
    last_loss: Optional[float] = None,
) -> None:
    """更新任务训练进度。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        current_step: 当前步数。
        max_steps: 总步数。
        last_loss: 最新损失值。
    """
    update_job_progress(
        db=db,
        job_id=job_id,
        current_step=current_step,
        max_steps=max_steps,
        last_loss=last_loss,
    )


def complete_job_training(
    db: Any,
    job_id: str,
    target_model_map: dict[str, str] | list[str],
) -> None:
    """标记任务为完成状态。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        target_model_map: 微调模型路径映射（兼容旧列表）。
    """
    if isinstance(target_model_map, list):
        target_model_map = {
            f"group_{idx + 1}": str(path)
            for idx, path in enumerate(target_model_map)
        }
    mark_job_completed(
        db=db,
        job_id=job_id,
        target_model_map=target_model_map,
        finished_at=datetime.now(timezone.utc),
    )


def fail_job_training(
    db: Any,
    job_id: str,
    error_message: str,
) -> None:
    """标记任务为失败状态。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        error_message: 错误信息。
    """
    mark_job_failed(
        db=db,
        job_id=job_id,
        error_message=error_message,
        finished_at=datetime.now(timezone.utc),
    )


def get_job_detail(db: Any, job_id: str) -> JobDetailResponse:
    """获取任务详情。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。

    Returns:
        任务详情响应。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    progress = JobProgressResponse(
        current_step=job.current_step,
        max_steps=job.max_steps,
        last_loss=job.last_loss,
    )

    target_model_map = _deserialize_target_model_map(job.target_model_map)
    model_paths = _deserialize_model_paths(job.model_paths)
    if model_paths is None and target_model_map:
        model_paths = list(target_model_map.values())
    if target_model_map is None:
        target_model_map = _build_target_model_map_fallback(job.request_json, model_paths)

    return JobDetailResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress=progress,
        error_message=job.error_message,
        log_path=job.log_path,
        model_paths=model_paths,
        target_model_map=target_model_map,
        output_dir=job.output_dir,
        metrics=_build_loss_metrics(db, job_id, job.request_json),
    )


def read_job_log(db: Any, job_id: str, tail: Optional[int] = None) -> str:
    """读取任务日志。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。
        tail: 仅返回最后 N 行（可选）。

    Returns:
        日志文本。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    if not job.log_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="日志路径不存在",
        )

    log_path = Path(job.log_path)
    if not log_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="日志文件不存在",
        )

    if tail is not None and tail <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tail 必须是正整数",
        )

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"读取日志失败: {exc}",
        ) from exc

    if tail is not None:
        lines = lines[-tail:]

    return "\n".join(lines)


def list_job_summaries(db: Any, limit: int = 20) -> JobListResponse:
    """获取任务列表摘要。

    Args:
        db: 数据库会话。
        limit: 返回的最大条数。

    Returns:
        任务列表响应。
    """
    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit 必须是正整数",
        )

    jobs = list_recent_jobs(db, limit=limit)
    items = [
        JobListItemResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
        for job in jobs
    ]

    return JobListResponse(items=items)


def list_job_summaries_with_status(
    db: Any,
    limit: int = 20,
    job_status: Optional[str] = None,
) -> JobListResponse:
    """获取任务列表摘要（支持按状态过滤）。"""
    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit 必须是正整数",
        )

    valid_status = {
        JobStatus.queued.value,
        JobStatus.running.value,
        JobStatus.completed.value,
    }

    if job_status is not None and job_status not in valid_status:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"status 必须是 {sorted(valid_status)} 之一",
        )

    if job_status is None:
        jobs = list_recent_jobs(db, limit=limit)
    else:
        jobs = list_recent_jobs_by_status(db, status=job_status, limit=limit)

    items = [
        JobListItemResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
        for job in jobs
    ]

    return JobListResponse(items=items)


def request_cancel_job(db: Any, job_id: str) -> CancelJobResponse:
    """请求取消任务。

    Args:
        db: 数据库会话。
        job_id: 任务 ID。

    Returns:
        取消任务响应。
    """
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    if job.status == JobStatus.queued.value:
        cancelled_job = mark_job_cancelled(db, job_id)
        assert cancelled_job, "job_id此时必然存在"
        return CancelJobResponse(
            job_id=cancelled_job.id,
            status=cancelled_job.status,
            cancel_requested=cancelled_job.cancel_requested,
            message="任务已取消（queued）",
        )

    if job.status == JobStatus.running.value:
        updated_job = set_cancel_requested(db, job_id, True)
        assert updated_job, "job_id此时必然存在"
        return CancelJobResponse(
            job_id=updated_job.id,
            status=updated_job.status,
            cancel_requested=updated_job.cancel_requested,
            message="已请求取消，等待训练停止",
        )

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f"当前状态无法取消: {job.status}",
    )


def _delete_job_local_files(job: Any) -> int:
    """删除任务相关本地产物（不包含发布目录）。"""
    deleted = 0

    output_dir = getattr(job, "output_dir", None)
    if output_dir:
        path = Path(output_dir)
        if path.exists() and path.is_dir():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            deleted += 1

    log_path = getattr(job, "log_path", None)
    if log_path:
        path = Path(log_path)
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
            deleted += 1

    return deleted


def _cancel_running_and_wait(
    db: Any,
    job_id: str,
    timeout_seconds: float = 15.0,
    poll_interval_seconds: float = 0.2,
) -> Any:
    """对 running 任务发起取消并等待其退出 running。"""
    set_cancel_requested(db, job_id, True)
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        db.expire_all()
        job = get_job_by_id(db, job_id)
        if job is None:
            return None
        if job.status != JobStatus.running.value:
            return job
        time.sleep(poll_interval_seconds)

    db.expire_all()
    return get_job_by_id(db, job_id)


def delete_single_job(db: Any, job_id: str) -> DeleteJobResponse:
    """删除单个任务。running 状态会先取消，待退出后再删除。"""
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {job_id}",
        )

    if job.status == JobStatus.running.value:
        job = _cancel_running_and_wait(db, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {job_id}",
            )
        if job.status == JobStatus.running.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="running 任务取消超时，请稍后重试删除",
            )

    queue = get_job_queue()
    removed_from_queue = queue.remove(job_id)
    files_deleted = _delete_job_local_files(job)
    deleted = delete_job_by_id(db, job_id)

    return DeleteJobResponse(
        job_id=job_id,
        deleted=deleted,
        removed_from_queue=removed_from_queue,
        files_deleted=files_deleted,
        message="任务已删除",
    )


def batch_delete_jobs(
    db: Any,
    job_status: Optional[str] = None,
    delete_all: bool = False,
) -> BatchDeleteJobsResponse:
    """批量删除任务，可按状态删除或全量删除。"""
    if delete_all and job_status is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="all=true 时不允许同时传 status",
        )
    if (not delete_all) and (job_status is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请传 status 或 all=true",
        )

    valid_status = {
        JobStatus.queued.value,
        JobStatus.running.value,
        JobStatus.completed.value,
        JobStatus.failed.value,
        JobStatus.cancelled.value,
    }
    if job_status is not None and job_status not in valid_status:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"status 必须是 {sorted(valid_status)} 之一",
        )

    candidates = list_jobs(db) if delete_all else list_jobs_by_status(db, job_status or "")
    queue = get_job_queue()

    matched_jobs = len(candidates)
    deleted_jobs = 0
    skipped_running_jobs = 0
    removed_from_queue = 0
    files_deleted = 0

    for job in candidates:
        if job.status == JobStatus.running.value:
            waited_job = _cancel_running_and_wait(db, job.id)
            if waited_job is None:
                continue
            if waited_job.status == JobStatus.running.value:
                skipped_running_jobs += 1
                continue
            job = waited_job

        if queue.remove(job.id):
            removed_from_queue += 1
        files_deleted += _delete_job_local_files(job)
        if delete_job_by_id(db, job.id):
            deleted_jobs += 1

    return BatchDeleteJobsResponse(
        matched_jobs=matched_jobs,
        deleted_jobs=deleted_jobs,
        skipped_running_jobs=skipped_running_jobs,
        removed_from_queue=removed_from_queue,
        files_deleted=files_deleted,
        message="批量删除完成",
    )
