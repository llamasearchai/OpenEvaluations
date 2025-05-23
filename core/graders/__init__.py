"""
Metric implementations and grading utilities
"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import logging
from functools import partial
import re
from difflib import SequenceMatcher

from openevals.config.data_structures import (
    MetricType, 
    GradingOutput, 
    MetricResult,
    TransformationType
)

logger = logging.getLogger(__name__)

# Registry for all available metrics
METRIC_REGISTRY: Dict[str, 'MetricFunction'] = {}

@dataclass
class MetricFunction:
    """Wrapper for metric functions with metadata"""
    name: str
    func: Callable
    metric_type: MetricType
    config: Dict[str, Any] = None

def exact_match(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
    """Exact string match metric"""
    passed = str(system_output).strip() == str(reference).strip()
    return {
        "value": float(passed),
        "passed": passed,
        "details": {
            "system_output": system_output,
            "reference": reference
        }
    }

def regex_match(system_output: Any, reference: Any, pattern: str, **kwargs) -> Dict[str, Any]:
    """Regex pattern match metric"""
    match = re.fullmatch(pattern, str(system_output))
    passed = match is not None
    return {
        "value": float(passed),
        "passed": passed,
        "details": {
            "pattern": pattern,
            "match": match.group() if match else None
        }
    }

def f1_metric(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
    """F1 score between tokenized outputs"""
    def _get_tokens(text):
        return re.findall(r'\w+', str(text).lower())
    
    sys_tokens = _get_tokens(system_output)
    ref_tokens = _get_tokens(reference)
    
    common = set(sys_tokens) & set(ref_tokens)
    precision = len(common) / len(sys_tokens) if sys_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "value": f1,
        "passed": None,  # No binary pass/fail for F1
        "details": {
            "precision": precision,
            "recall": recall,
            "common_tokens": list(common)
        }
    }

def semantic_similarity(system_output: Any, reference: Any, **kwargs) -> Dict[str, Any]:
    """Semantic similarity using difflib (basic implementation)"""
    similarity = SequenceMatcher(None, str(system_output), str(reference)).ratio()
    return {
        "value": similarity,
        "passed": None,  # No binary pass/fail
        "details": {
            "similarity": similarity
        }
    }

def register_grader(name: str, func: Callable, metric_type: MetricType) -> MetricFunction:
    """Register a new grading metric"""
    if name in METRIC_REGISTRY:
        logger.warning(f"Overwriting existing metric: {name}")
    
    metric = MetricFunction(name=name, func=func, metric_type=metric_type)
    METRIC_REGISTRY[name] = metric
    return metric

def get_grader(name: str) -> Optional[MetricFunction]:
    """Get a registered grading metric"""
    return METRIC_REGISTRY.get(name)

def apply_transformations(value: Any, transformations: List[Dict[str, Any]]) -> Any:
    """Apply transformations to a value"""
    result = value
    for transform in transformations:
        transform_type = transform.get('transform_type')
        config = transform.get('config', {})
        
        if transform_type == TransformationType.TO_LOWERCASE:
            result = str(result).lower()
        elif transform_type == TransformationType.STRIP_WHITESPACE:
            result = str(result).strip()
        elif transform_type == TransformationType.REGEX_EXTRACT:
            pattern = config.get('pattern')
            if pattern:
                match = re.search(pattern, str(result))
                result = match.group() if match else ""
        # Add more transformation types as needed
        
    return result

def grade_case(
    system_output: Any,
    reference: Any,
    metrics: List[Dict[str, Any]],
    common_transformations: Optional[List[Dict[str, Any]]] = None
) -> GradingOutput:
    """Grade a system output against a reference using multiple metrics"""
    if common_transformations is None:
        common_transformations = []
    
    # Apply common transformations
    processed_sys = apply_transformations(system_output, common_transformations)
    processed_ref = apply_transformations(reference, common_transformations)
    
    metric_results = []
    all_passed = True
    
    for metric_def in metrics:
        metric_name = metric_def['name']
        metric = get_grader(metric_name)
        if not metric:
            logger.warning(f"Unknown metric: {metric_name}")
            continue
            
        try:
            # Apply metric-specific transformations if any
            metric_transforms = metric_def.get('transformations', [])
            metric_sys = apply_transformations(processed_sys, metric_transforms)
            metric_ref = apply_transformations(processed_ref, metric_transforms)
            
            # Run the metric
            result = metric.func(
                system_output=metric_sys,
                reference=metric_ref,
                **metric_def.get('config', {})
            )
            
            # Check against threshold if specified
            threshold = metric_def.get('threshold')
            passed = result.get('passed', None)
            if passed is None and threshold is not None:
                passed = result['value'] >= threshold
            
            metric_results.append(MetricResult(
                metric_name=metric_name,
                metric_type=metric.metric_type,
                value=result['value'],
                details=result.get('details'),
                threshold=threshold,
                passed=passed
            ))
            
            if passed is False:
                all_passed = False
                if metric_def.get('stop_on_failure', False):
                    break
                    
        except Exception as e:
            logger.error(f"Metric {metric_name} failed: {str(e)}")
            metric_results.append(MetricResult(
                metric_name=metric_name,
                metric_type=metric.metric_type,
                value=0.0,
                details={"error": str(e)},
                passed=False
            ))
            all_passed = False
    
    # Calculate overall score (average of metric values)
    if metric_results:
        overall_score = sum(mr.value for mr in metric_results) / len(metric_results)
    else:
        overall_score = 0.0
    
    return GradingOutput(
        passed=all_passed,
        score=overall_score,
        metric_results=metric_results,
        raw_output_system=system_output,
        raw_output_reference=reference
    )

# Register built-in metrics
register_grader("exact_match", exact_match, MetricType.EXACT_MATCH)
register_grader("regex_match", regex_match, MetricType.REGEX_MATCH)
register_grader("f1_score", f1_metric, MetricType.F1_SCORE)
register_grader("semantic_similarity", semantic_similarity, MetricType.SEMANTIC_SIMILARITY) 