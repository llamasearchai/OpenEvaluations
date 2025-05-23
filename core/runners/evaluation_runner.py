"""
Advanced Evaluation Runner for OpenEvals
========================================

This module provides a robust, production-ready evaluation runner with:
- Parallel execution with configurable workers
- Progress tracking and real-time updates
- Comprehensive error handling and retry mechanisms
- Result aggregation and reporting
- Multi-modal support
- Performance monitoring

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
import json
import traceback

from openevals.core.definitions import EvalSuiteRun, EvalTask, EvalCase
from openevals.core.adapters import get_adapter, AbstractAdapter
from openevals.core.graders import get_grader, apply_transformations
from openevals.core.quality_checks import run_quality_checks
from openevals.core.reporting import generate_json_report, generate_console_report, generate_html_report
from openevals.config import GlobalConfig, EvalSuiteConfig, TargetSystemConfig
from openevals.config.data_structures import EvalRunStatus, MetricResult, GradingOutput

logger = logging.getLogger(__name__)

class EvaluationProgress:
    """Tracks progress of evaluation runs"""
    
    def __init__(self, total_cases: int):
        self.total_cases = total_cases
        self.completed_cases = 0
        self.failed_cases = 0
        self.start_time = time.time()
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add progress callback function"""
        self.callbacks.append(callback)
    
    def update(self, case_result: str):
        """Update progress with case result"""
        if case_result == "completed":
            self.completed_cases += 1
        elif case_result == "failed":
            self.failed_cases += 1
        
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_cases == 0:
            return 100.0
        return (self.completed_cases + self.failed_cases) / self.total_cases * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining(self) -> float:
        """Estimate remaining time in seconds"""
        if self.completed_cases == 0:
            return 0.0
        rate = self.completed_cases / self.elapsed_time
        remaining_cases = self.total_cases - (self.completed_cases + self.failed_cases)
        return remaining_cases / rate if rate > 0 else 0.0

class EvaluationRunner:
    """
    Advanced evaluation runner with parallel execution and monitoring
    """
    
    def __init__(
        self,
        global_config: GlobalConfig,
        max_workers: int = 4,
        retry_attempts: int = 3,
        timeout_seconds: int = 300,
        enable_quality_checks: bool = True
    ):
        self.global_config = global_config
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.timeout_seconds = timeout_seconds
        self.enable_quality_checks = enable_quality_checks
        self.adapters_cache: Dict[str, AbstractAdapter] = {}
        
    def run_suite(
        self,
        suite_config: EvalSuiteConfig,
        target_system: TargetSystemConfig,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None
    ) -> EvalSuiteRun:
        """
        Run a complete evaluation suite
        
        Args:
            suite_config: Configuration for the evaluation suite
            target_system: Target system to evaluate
            output_dir: Directory to save results
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvalSuiteRun object with results
        """
        run_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        logger.info(f"Starting evaluation run {run_id} for suite {suite_config.suite_id}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(self.global_config.default_output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize suite run
        suite_run = EvalSuiteRun(
            suite_config=suite_config,
            tasks=[],
            target_system_config=target_system,
            run_id=run_id,
            start_time=start_time.isoformat(),
            status=EvalRunStatus.RUNNING,
            output_dir=str(output_dir)
        )
        
        try:
            # Load tasks
            tasks = self._load_tasks(suite_config)
            suite_run.tasks = tasks
            
            # Count total cases for progress tracking
            total_cases = sum(len(task.cases) for task in tasks)
            progress = EvaluationProgress(total_cases)
            
            if progress_callback:
                progress.add_callback(progress_callback)
            
            # Get adapter for target system
            adapter = self._get_adapter(target_system)
            
            # Run tasks
            if self.max_workers == 1:
                self._run_tasks_sequential(tasks, adapter, progress)
            else:
                self._run_tasks_parallel(tasks, adapter, progress)
            
            # Calculate overall results
            self._calculate_suite_metrics(suite_run)
            
            # Quality checks
            if self.enable_quality_checks:
                self._run_suite_quality_checks(suite_run)
            
            suite_run.status = EvalRunStatus.COMPLETED
            logger.info(f"Evaluation run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation run {run_id} failed: {str(e)}")
            suite_run.status = EvalRunStatus.FAILED
            # Save error details
            error_file = output_dir / "error.json"
            with open(error_file, 'w') as f:
                json.dump({
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        
        finally:
            suite_run.end_time = datetime.now(timezone.utc).isoformat()
            
            # Generate reports
            try:
                self._generate_reports(suite_run, output_dir)
            except Exception as e:
                logger.error(f"Failed to generate reports: {str(e)}")
        
        return suite_run
    
    def _load_tasks(self, suite_config: EvalSuiteConfig) -> List[EvalTask]:
        """Load tasks from suite configuration"""
        tasks = []
        
        for task_ref in suite_config.tasks:
            try:
                # Load task definition
                if task_ref.source_type.value == "local_file":
                    task = self._load_task_from_file(task_ref.source_path)
                elif task_ref.source_type.value == "registry_reference":
                    task = self._load_task_from_registry(task_ref.source_path)
                else:
                    raise ValueError(f"Unsupported source type: {task_ref.source_type}")
                
                # Apply overrides
                if task_ref.overrides:
                    task = self._apply_task_overrides(task, task_ref.overrides)
                
                tasks.append(task)
                
            except Exception as e:
                logger.error(f"Failed to load task {task_ref.task_id}: {str(e)}")
                # Create a failed task placeholder
                failed_task = EvalTask(
                    task_id=task_ref.task_id,
                    default_grading_criteria=None,
                    cases=[],
                    status=EvalRunStatus.FAILED
                )
                tasks.append(failed_task)
        
        return tasks
    
    def _load_task_from_file(self, file_path: str) -> EvalTask:
        """Load task from YAML/JSON file"""
        import yaml
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {file_path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                task_data = yaml.safe_load(f)
            else:
                task_data = json.load(f)
        
        return EvalTask.from_dict(task_data)
    
    def _load_task_from_registry(self, registry_ref: str) -> EvalTask:
        """Load task from registry"""
        # Implementation would depend on registry system
        raise NotImplementedError("Registry loading not yet implemented")
    
    def _apply_task_overrides(self, task: EvalTask, overrides: Dict[str, Any]) -> EvalTask:
        """Apply configuration overrides to a task"""
        # Create a copy and apply overrides
        task_dict = asdict(task)
        
        # Apply nested overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like "grading.threshold"
                keys = key.split('.')
                current = task_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                task_dict[key] = value
        
        return EvalTask.from_dict(task_dict)
    
    def _get_adapter(self, target_system: TargetSystemConfig) -> AbstractAdapter:
        """Get or create adapter for target system"""
        if target_system.name in self.adapters_cache:
            return self.adapters_cache[target_system.name]
        
        adapter = get_adapter(target_system.adapter_type, target_system.adapter_config)
        self.adapters_cache[target_system.name] = adapter
        return adapter
    
    def _run_tasks_sequential(
        self,
        tasks: List[EvalTask],
        adapter: AbstractAdapter,
        progress: EvaluationProgress
    ):
        """Run tasks sequentially"""
        for task in tasks:
            self._run_task(task, adapter, progress)
    
    def _run_tasks_parallel(
        self,
        tasks: List[EvalTask],
        adapter: AbstractAdapter,
        progress: EvaluationProgress
    ):
        """Run tasks in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_task, task, adapter, progress): task
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {str(e)}")
                    task.status = EvalRunStatus.FAILED
    
    def _run_task(
        self,
        task: EvalTask,
        adapter: AbstractAdapter,
        progress: EvaluationProgress
    ):
        """Run a single evaluation task"""
        logger.info(f"Running task: {task.task_id}")
        task.status = EvalRunStatus.RUNNING
        
        start_time = time.time()
        
        try:
            # Run each case in the task
            for case in task.cases:
                self._run_case(case, task, adapter, progress)
            
            # Calculate task-level metrics
            self._calculate_task_metrics(task)
            
            task.status = EvalRunStatus.COMPLETED
            logger.info(f"Task {task.task_id} completed")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            task.status = EvalRunStatus.FAILED
        
        finally:
            task.duration_seconds = time.time() - start_time
    
    def _run_case(
        self,
        case: EvalCase,
        task: EvalTask,
        adapter: AbstractAdapter,
        progress: EvaluationProgress
    ):
        """Run a single evaluation case with retry logic"""
        logger.debug(f"Running case: {case.case_id}")
        case.status = EvalRunStatus.RUNNING
        
        start_time = time.time()
        
        for attempt in range(self.retry_attempts):
            try:
                # Prepare inputs
                prepared_inputs = self._prepare_inputs(case, task)
                
                # Run inference
                system_output = adapter.run_inference(prepared_inputs, timeout=self.timeout_seconds)
                case.system_output = system_output
                
                # Apply transformations to system output
                case.processed_system_output = self._apply_output_transformations(
                    system_output, case
                )
                
                # Grade the case
                grading_output = self._grade_case(case, task)
                case.grading_output = grading_output
                
                case.status = EvalRunStatus.COMPLETED
                progress.update("completed")
                break
                
            except Exception as e:
                logger.warning(f"Case {case.case_id} attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    # Final attempt failed
                    case.status = EvalRunStatus.FAILED
                    case.error_message = str(e)
                    progress.update("failed")
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        case.duration_seconds = time.time() - start_time
    
    def _prepare_inputs(self, case: EvalCase, task: EvalTask) -> Dict[str, Any]:
        """Prepare inputs for the adapter"""
        prepared = {}
        
        for eval_input in case.inputs:
            # Apply transformations
            value = eval_input.value
            for transform in eval_input.transformations:
                value = apply_transformations(value, [transform])
            
            prepared[eval_input.name] = value
        
        # Add task instruction if available
        if task.instruction_prompt:
            prepared['instruction'] = task.instruction_prompt
        
        return prepared
    
    def _apply_output_transformations(self, output: Any, case: EvalCase) -> Any:
        """Apply transformations to system output"""
        # Apply common transformations from references
        for reference in case.references:
            for transform in reference.common_transformations:
                output = apply_transformations(output, [transform])
        
        return output
    
    def _grade_case(self, case: EvalCase, task: EvalTask) -> GradingOutput:
        """Grade a single evaluation case"""
        # Use case-specific grading criteria or task default
        criteria = case.grading_criteria or task.default_grading_criteria
        
        if not criteria:
            raise ValueError(f"No grading criteria found for case {case.case_id}")
        
        metric_results = []
        all_passed = True
        total_score = 0.0
        total_weight = 0.0
        
        for metric_def in criteria.metrics:
            try:
                # Get the grader function
                grader = get_grader(metric_def.metric_type.value)
                
                # Prepare grading inputs
                system_output = case.processed_system_output
                reference_output = case.references[0].value if case.references else None
                
                # Run grading
                result = grader(
                    system_output=system_output,
                    reference=reference_output,
                    **metric_def.config
                )
                
                # Create metric result
                metric_result = MetricResult(
                    metric_name=metric_def.name,
                    metric_type=metric_def.metric_type,
                    value=result.get('value', 0.0),
                    details=result.get('details', {}),
                    threshold=metric_def.pass_threshold,
                    passed=result.get('passed')
                )
                
                # Check threshold if specified
                if metric_def.pass_threshold is not None and metric_result.passed is None:
                    if isinstance(metric_result.value, (int, float)):
                        metric_result.passed = metric_result.value >= metric_def.pass_threshold
                    else:
                        metric_result.passed = metric_result.value == metric_def.pass_threshold
                
                metric_results.append(metric_result)
                
                # Update overall scoring
                if isinstance(metric_result.value, (int, float)):
                    total_score += metric_result.value * metric_def.weight
                    total_weight += metric_def.weight
                
                # Update pass status
                if metric_result.passed is False:
                    all_passed = False
                    if criteria.stop_on_first_failure:
                        break
                        
            except Exception as e:
                logger.error(f"Grading failed for metric {metric_def.name}: {str(e)}")
                metric_result = MetricResult(
                    metric_name=metric_def.name,
                    metric_type=metric_def.metric_type,
                    value=0.0,
                    passed=False,
                    details={"error": str(e)}
                )
                metric_results.append(metric_result)
                all_passed = False
        
        # Calculate overall score
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Apply aggregation strategy
        if criteria.aggregation_strategy == "all_must_pass":
            passed = all_passed
        elif criteria.aggregation_strategy == "weighted_average_above_threshold":
            threshold = criteria.aggregation_config.get("threshold", 0.5)
            passed = overall_score >= threshold
        else:
            passed = all_passed  # Default fallback
        
        return GradingOutput(
            passed=passed,
            score=overall_score,
            metric_results=metric_results,
            raw_output_system=case.system_output,
            raw_output_reference=case.references[0].value if case.references else None
        )
    
    def _calculate_task_metrics(self, task: EvalTask):
        """Calculate aggregated metrics for a task"""
        if not task.cases:
            return
        
        completed_cases = [c for c in task.cases if c.status == EvalRunStatus.COMPLETED]
        if not completed_cases:
            task.overall_score = 0.0
            return
        
        # Calculate pass rate
        passed_cases = [c for c in completed_cases if c.grading_output and c.grading_output.passed]
        pass_rate = len(passed_cases) / len(completed_cases)
        
        # Calculate average score
        scores = [c.grading_output.score for c in completed_cases if c.grading_output]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        task.overall_score = avg_score
        task.summary_metrics = {
            "pass_rate": pass_rate,
            "average_score": avg_score,
            "total_cases": len(task.cases),
            "completed_cases": len(completed_cases),
            "passed_cases": len(passed_cases),
            "failed_cases": len([c for c in task.cases if c.status == EvalRunStatus.FAILED])
        }
    
    def _calculate_suite_metrics(self, suite_run: EvalSuiteRun):
        """Calculate aggregated metrics for the entire suite"""
        if not suite_run.tasks:
            return
        
        completed_tasks = [t for t in suite_run.tasks if t.status == EvalRunStatus.COMPLETED]
        if not completed_tasks:
            suite_run.overall_score = 0.0
            return
        
        # Calculate weighted average score
        total_score = 0.0
        total_cases = 0
        
        for task in completed_tasks:
            if task.overall_score is not None:
                task_cases = len(task.cases)
                total_score += task.overall_score * task_cases
                total_cases += task_cases
        
        suite_run.overall_score = total_score / total_cases if total_cases > 0 else 0.0
        
        # Aggregate summary metrics
        suite_run.summary_metrics = {
            "overall_score": suite_run.overall_score,
            "total_tasks": len(suite_run.tasks),
            "completed_tasks": len(completed_tasks),
            "total_cases": sum(len(t.cases) for t in suite_run.tasks),
            "task_scores": {t.task_id: t.overall_score for t in completed_tasks}
        }
    
    def _run_suite_quality_checks(self, suite_run: EvalSuiteRun):
        """Run quality checks on the suite results"""
        for task in suite_run.tasks:
            try:
                run_quality_checks(task, strict=False)
            except Exception as e:
                logger.warning(f"Quality check failed for task {task.task_id}: {str(e)}")
    
    def _generate_reports(self, suite_run: EvalSuiteRun, output_dir: Path):
        """Generate evaluation reports"""
        try:
            # JSON report
            json_path = generate_json_report(suite_run, str(output_dir))
            logger.info(f"JSON report generated: {json_path}")
            
            # Console report
            generate_console_report(suite_run)
            
            # HTML report
            html_path = generate_html_report(suite_run, str(output_dir))
            logger.info(f"HTML report generated: {html_path}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")

# Async version for high-performance scenarios
class AsyncEvaluationRunner(EvaluationRunner):
    """Async version of EvaluationRunner for high-performance scenarios"""
    
    async def run_suite_async(
        self,
        suite_config: EvalSuiteConfig,
        target_system: TargetSystemConfig,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None
    ) -> EvalSuiteRun:
        """Async version of run_suite"""
        # Convert sync operations to async where beneficial
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run_suite, suite_config, target_system, output_dir, progress_callback
        ) 