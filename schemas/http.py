from typing import List, Union, Optional, Literal
from pydantic import BaseModel

# v1/embeddings request and response schemas
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
