"""
Defines the core data structures for representing evaluations, tasks, cases, 
metrics, and their results within the Open-Evals framework.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable

from open_evals.config.data_structures import (
    MetricType, InputFormat, OutputType, TransformationType, 
    GradingOutput, MetricResult, TargetSystemConfig, EvalRunStatus
)
from open_evals.config.config import EvalSuiteConfig, EvalTaskConfigRef # For type hinting

@dataclass
class MetricDefinition:
    """Definition of a metric to be applied during grading."""
    name: str
    metric_type: MetricType
    # Configuration specific to the metric type, e.g., regex pattern, LLM judge prompt
    config: Dict[str, Any] = field(default_factory=dict)
    # Optional: Weight of this metric in an overall score if multiple metrics are used
    weight: float = 1.0
    # Optional: Threshold for this metric to be considered passed
    pass_threshold: Optional[Union[float, int, bool, str]] = None 

@dataclass
class GradingCriteria:
    """Defines how an evaluation case or task is graded."""
    # List of metrics to apply
    metrics: List[MetricDefinition]
    # Strategy for combining metric results into an overall pass/fail or score
    # e.g., "all_must_pass", "weighted_average_above_threshold", "primary_metric_passes"
    aggregation_strategy: str = "all_must_pass"
    # Config for the aggregation strategy, e.g., threshold for weighted_average
    aggregation_config: Dict[str, Any] = field(default_factory=dict)
    # If true, stop grading further metrics for a case if one fails
    stop_on_first_failure: bool = False

@dataclass
class TransformationStep:
    """Defines a single transformation step for input or output data."""
    transform_type: TransformationType
    config: Dict[str, Any] = field(default_factory=dict) # e.g., regex for REGEX_EXTRACT

@dataclass
class EvalInput:
    """Represents the input for a single evaluation case."""
    name: str # Optional name for the input field if multiple inputs
    value: Any # The actual input data
    input_format: InputFormat = InputFormat.TEXT
    # Optional transformations to apply to this input before sending to the system
    transformations: List[TransformationStep] = field(default_factory=list)

@dataclass
class EvalOutputReference:
    """Represents the reference (ground truth) output for an evaluation case."""
    value: Any # The ground truth value
    output_type: OutputType = OutputType.TEXT
    # Optional transformations to apply to reference and system output before grading
    # (applied to both to ensure fair comparison)
    common_transformations: List[TransformationStep] = field(default_factory=list)
    # Optional metadata for the reference, e.g., source, difficulty level
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvalCase:
    """Represents a single test case within an evaluation task."""
    case_id: str # Unique identifier for the case within its task
    description: Optional[str] = None
    inputs: List[EvalInput]
    references: List[EvalOutputReference] # Could be multiple references for some tasks
    # Specific grading criteria for this case, overrides task-level if provided
    grading_criteria: Optional[GradingCriteria] = None 
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., difficulty, source
    # Results after running and grading this case
    status: EvalRunStatus = EvalRunStatus.PENDING
    system_output: Optional[Any] = None # Raw output from the system under test
    processed_system_output: Optional[Any] = None # Output after transformations
    grading_output: Optional[GradingOutput] = None
    error_message: Optional[str] = None # If the case failed to run or grade
    duration_seconds: Optional[float] = None # Time taken to run this case

@dataclass
class EvalTask:
    """Represents an evaluation task, composed of multiple evaluation cases."""
    task_id: str # Unique identifier for the task
    description: Optional[str] = None
    # Default grading criteria for all cases in this task (can be overridden per case)
    default_grading_criteria: GradingCriteria
    # List of evaluation cases
    cases: List[EvalCase]
    tags: List[str] = field(default_factory=list)
    version: Optional[str] = "1.0"
    # Instructions or prompt template for the task, if applicable
    # This can be used by runners to format inputs for the target system
    instruction_prompt: Optional[str] = None
    # Metadata for the task
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Aggregated results for the task
    status: EvalRunStatus = EvalRunStatus.PENDING
    overall_score: Optional[float] = None
    summary_metrics: Dict[str, Union[float, str, Dict]] = field(default_factory=dict)
    # Path from where this task definition was loaded (if applicable)
    source_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalTask":
        """Create an EvalTask from a dictionary (e.g., loaded from YAML/JSON)."""
        # This is a simplified loader. A more robust one would handle nested dataclasses.
        cases_data = data.pop("cases", [])
        cases = []
        for case_data in cases_data:
            inputs_data = case_data.pop("inputs", [])
            inputs = [EvalInput(**inp) for inp in inputs_data]
            references_data = case_data.pop("references", [])
            references = [EvalOutputReference(**ref) for ref in references_data]
            
            criteria_data = case_data.pop("grading_criteria", None)
            criteria = GradingCriteria(**criteria_data) if criteria_data else None
            # TODO: Parse metrics within criteria properly
            
            cases.append(EvalCase(inputs=inputs, references=references, grading_criteria=criteria, **case_data))

        default_criteria_data = data.pop("default_grading_criteria")
        metrics_def_data = default_criteria_data.pop("metrics", [])
        metrics_defs = [MetricDefinition(**md) for md in metrics_def_data]
        default_criteria = GradingCriteria(metrics=metrics_defs, **default_criteria_data)
        
        return cls(cases=cases, default_grading_criteria=default_criteria, **data)

@dataclass
class EvalSuiteRun:
    """Represents an instance of an evaluation suite being run or its results."""
    suite_config: EvalSuiteConfig # The configuration that defined this suite run
    tasks: List[EvalTask] # The fully resolved EvalTask objects for this suite
    target_system_config: TargetSystemConfig # The specific system this suite is run against
    run_id: str # Unique ID for this particular run of the suite
    start_time: Optional[str] = None # ISO 8601 timestamp
    end_time: Optional[str] = None # ISO 8601 timestamp
    status: EvalRunStatus = EvalRunStatus.PENDING
    # Aggregated results for the suite
    overall_score: Optional[float] = None
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    # Path to where detailed results and reports for this run are stored
    output_dir: Optional[str] = None

# Placeholder for a class that manages loading EvalTasks from files or registry
class EvalTaskLoader: # pragma: no cover
    def __init__(self, registry_path: Optional[str] = None):
        self.registry: Dict[str, Any] = {}
        if registry_path:
            # TODO: Implement loading registry from YAML/JSON
            pass

    def load_task(self, task_ref: EvalTaskConfigRef) -> EvalTask:
        """Loads an EvalTask based on its reference."""
        if task_ref.source_type == StoredEvalType.LOCAL_FILE:
            if not task_ref.source_path:
                raise ValueError(f"source_path is required for local_file task: {task_ref.task_id}")
            # TODO: Implement loading EvalTask from YAML/JSON file
            # For now, returning a dummy task
            dummy_metric_def = MetricDefinition(name="dummy_exact_match", metric_type=MetricType.EXACT_MATCH)
            dummy_criteria = GradingCriteria(metrics=[dummy_metric_def])
            dummy_case = EvalCase(case_id="dummy_case_1", inputs=[], references=[])
            return EvalTask(task_id=task_ref.task_id, default_grading_criteria=dummy_criteria, cases=[dummy_case], source_path=task_ref.source_path)
        elif task_ref.source_type == StoredEvalType.REGISTRY_REFERENCE:
            # TODO: Implement loading from registry
            raise NotImplementedError("Registry loading not yet implemented.")
        else:
            raise ValueError(f"Unsupported source_type: {task_ref.source_type}") 