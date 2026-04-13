"""
工具接口。
"""

from io import StringIO

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter(prefix="/v1/tools", tags=["tools"])


class CorrelationRequest(BaseModel):
    """计算相关性矩阵的请求。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "csv_content": "value1,value2,value3\n1,2,3\n4,5,6\n",
                "columns": ["value1", "value2"],
                "method": "pearson",
            }
        }
    )

    csv_content: str = Field(
        description="CSV 数据内容，必须包含表头行",
    )
    columns: list[str] = Field(
        description="用于计算相关性的列名列表",
    )
    method: str = Field(
        default="pearson",
        description="相关性计算方法: 'pearson', 'spearman', 或 'kendall'",
    )

    @field_validator("csv_content")
    @classmethod
    def validate_csv_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("csv_content 不能为空")
        return v

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("columns 不能为空")
        if any(not item or not item.strip() for item in v):
            raise ValueError("columns 不能包含空值")
        return v

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """验证相关性计算方法。"""
        valid_methods = {"pearson", "spearman", "kendall"}
        if v not in valid_methods:
            raise ValueError(f"method 必须是 {valid_methods} 之一，得到 {v}")
        return v


class CorrelationResponse(BaseModel):
    """相关性矩阵响应。"""

    correlation_matrix: dict[str, dict[str, float | None]] = Field(
        description="相关性矩阵，值为 Pearson 相关系数或 null",
    )


@router.post("/correlation", response_model=CorrelationResponse)
async def calculate_correlation(request: CorrelationRequest) -> CorrelationResponse:
    """根据传入的 CSV 数据和指定列计算相关性矩阵。"""
    try:
        dataframe = pd.read_csv(StringIO(request.csv_content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法解析 csv_content: {exc}",
        )

    missing_columns = [column for column in request.columns if column not in dataframe.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"找不到列: {missing_columns}",
        )

    try:
        numeric_data = dataframe[request.columns].apply(pd.to_numeric, errors="coerce")
        correlation = numeric_data.corr(method=request.method)
        correlation_matrix: dict[str, dict[str, float | None]] = correlation.where(~correlation.isna(), None).to_dict()  # type: ignore[assignment]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"相关性计算失败: {exc}",
        )

    return CorrelationResponse(correlation_matrix=correlation_matrix)
