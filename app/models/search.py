"""
Pydantic models for search request / response
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import settings


class SortMode(str, Enum):
    hybrid = "hybrid"
    hybrid_vector = "hybrid_vector"
    relevance = "relevance"
    date = "date"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    sort: SortMode = SortMode.hybrid
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default_factory=lambda: settings.DEFAULT_PAGE_SIZE,
        ge=1,
        description="Results per page",
    )
    indices: Optional[str] = Field(None, description="Comma-separated index names; defaults to DEFAULT_INDEX")
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    include_highlights: bool = True
    debug_scores: bool = False


class ScoreBreakdown(BaseModel):
    final_score: float
    bm25_score: float
    vector_score: Optional[float] = None
    source_rank: Optional[float] = None


class SearchResultItem(BaseModel):
    id: str
    title: str
    url: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[str] = None
    summary: Optional[str] = None
    highlights: Optional[dict] = None
    score: float
    score_breakdown: Optional[ScoreBreakdown] = None


class SearchResponse(BaseModel):
    query: str
    sort_mode: str
    total: int
    page: int
    page_size: int
    total_pages: int
    took_ms: int
    results: list[SearchResultItem]
