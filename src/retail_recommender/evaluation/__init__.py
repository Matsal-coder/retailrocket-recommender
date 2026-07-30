"""Recommendation evaluation utilities."""

from retail_recommender.evaluation.evaluator import (
    EvaluationResult,
    RecommendationProvider,
    RecommenderEvaluator,
)
from retail_recommender.evaluation.ranking_metrics import (
    coverage_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from retail_recommender.evaluation.reports import (
    write_evaluation_report,
    write_model_comparison,
)

__all__ = [
    "EvaluationResult",
    "RecommendationProvider",
    "RecommenderEvaluator",
    "coverage_at_k",
    "map_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "write_evaluation_report",
    "write_model_comparison",
]
