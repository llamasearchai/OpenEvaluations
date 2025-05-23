"""
Defines core data structures, enums, and constants for Open-Evals.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

class EvalRunStatus(Enum):
    """Status of an evaluation run or individual test case."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial" # For suites or tasks with mixed results

class MetricType(Enum):
    """Type of metric used for grading."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    ROUGE_L = "rouge_l"
    BLEU = "bleu"
    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    SEMANTIC_SIMILARITY = "semantic_similarity" # Requires embedding models
    LLM_AS_JUDGE = "llm_as_judge" # Requires an LLM call for grading
    CUSTOM = "custom" # For user-defined metrics

class InputFormat(Enum):
    """Format of the input data for an evaluation task."""
    TEXT = "text"
    JSON = "json"
    FILE_PATH = "file_path"
    # Engineering specific formats could be added, e.g., CAD, CSV_TIMESERIES

class OutputType(Enum):
    """Expected type of output from the system under test."""
    TEXT = "text"
    JSON = "json"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    # Engineering specific: e.g., COORDINATES, PARAMETER_SET

class TransformationType(Enum):
    """Type of transformation to apply to inputs or outputs."""
    NONE = "none"
    TO_LOWERCASE = "to_lowercase"
    STRIP_WHITESPACE = "strip_whitespace"
    JSON_PARSE = "json_parse"
    REGEX_EXTRACT = "regex_extract"
    # More complex: e.g., UNIT_CONVERSION, CODE_FORMATTING

class StoredEvalType(Enum):
    """Indicates where the eval definition is stored or how it's referenced."""
    LOCAL_FILE = "local_file"       # Defined in a local YAML/JSON file
    REGISTRY_REFERENCE = "registry_reference" # Points to an eval in a known registry
    DATABASE_ID = "database_id"     # Stored in a results DB and referenced by ID

@dataclass
class MetricResult:
    """Represents the result of a single metric calculation."""
    metric_name: str
    metric_type: MetricType
    value: Union[float, int, bool, str]
    details: Optional[Dict[str, Any]] = None # For richer context, e.g., confusion matrix
    threshold: Optional[float] = None # Optional pass/fail threshold for this metric
    passed: Optional[bool] = None # If a threshold is set

@dataclass
class GradingOutput:
    """Output of a grading process for a single evaluation case."""
    passed: bool # Overall pass/fail for the case based on all metrics
    score: float # Overall score (e.g., average of metric values, or primary metric)
    metric_results: List[MetricResult]
    # Optional fields for qualitative feedback or LLM-as-judge reasoning
    feedback: Optional[str] = None
    raw_output_system: Optional[Any] = None
    raw_output_reference: Optional[Any] = None

@dataclass
class TargetSystemConfig:
    """Configuration for the system under test."""
    name: str
    adapter_type: str # e.g., "openai_chat", "local_hf_pipeline", "custom_api"
    adapter_config: Dict[str, Any] = field(default_factory=dict) # API keys, model names, endpoints
    # Optional: Versioning information for the system
    version: Optional[str] = None

# Further structures for EvalCase, EvalTask, EvalSuite will be in core.definitions 