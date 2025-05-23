"""
Quality checks for evaluation tasks and results
"""
from typing import Dict, Any, List, Optional
import logging
from dataclasses import is_dataclass, asdict

from openevals.core.definitions import EvalTask, EvalCase

logger = logging.getLogger(__name__)

class QualityCheckError(Exception):
    """Exception raised when quality checks fail"""
    pass

def validate_task_definition(task: EvalTask) -> List[str]:
    """Validate an evaluation task definition"""
    errors = []
    
    # Check required fields
    if not task.task_id:
        errors.append("Task ID is required")
    if not task.cases:
        errors.append("Task must contain at least one evaluation case")
    
    # Validate each case
    for case in task.cases:
        case_errors = validate_case_definition(case)
        if case_errors:
            errors.extend(f"Case {case.case_id}: {e}" for e in case_errors)
    
    return errors

def validate_case_definition(case: EvalCase) -> List[str]:
    """Validate an evaluation case definition"""
    errors = []
    
    if not case.case_id:
        errors.append("Case ID is required")
    if not case.inputs:
        errors.append("Case must have at least one input")
    if not case.references:
        errors.append("Case must have at least one reference output")
    
    return errors

def check_result_consistency(task: EvalTask) -> List[str]:
    """Check for consistency in evaluation results"""
    warnings = []
    
    # Check if all cases were executed
    executed_cases = [c for c in task.cases if c.status not in ("pending", "skipped")]
    if len(executed_cases) != len(task.cases):
        warnings.append(f"Only {len(executed_cases)} of {len(task.cases)} cases were executed")
    
    # Check for failed cases
    failed_cases = [c for c in task.cases if c.status == "failed"]
    if failed_cases:
        warnings.append(f"{len(failed_cases)} cases failed during execution")
    
    return warnings

def run_quality_checks(task: EvalTask, strict: bool = False) -> None:
    """Run all quality checks on a task"""
    definition_errors = validate_task_definition(task)
    if definition_errors:
        error_msg = "Task definition errors:\n" + "\n".join(definition_errors)
        if strict:
            raise QualityCheckError(error_msg)
        logger.warning(error_msg)
    
    consistency_warnings = check_result_consistency(task)
    if consistency_warnings:
        logger.warning("Result consistency issues:\n" + "\n".join(consistency_warnings)) 