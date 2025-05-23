"""
OpenEvals: AI Evaluation Framework
==================================

OpenEvals is a comprehensive, modular framework designed for robustly evaluating
AI systems across various tasks and metrics. It supports configurable evaluation
suites, diverse AI system adapters, a rich set of grading metrics, and
flexible reporting capabilities.

Key Features:
- Define evaluation tasks and suites using YAML configuration.
- Interface with different AI models (OpenAI, Hugging Face, etc.) via adapters.
- Apply a wide range of metrics (exact match, F1, ROUGE, semantic similarity, LLM-as-judge).
- Generate detailed reports in JSON, console, and HTML formats.
- Run evaluations via a Command Line Interface (CLI) or a FastAPI-based API.
- Extensible design for adding custom adapters, metrics, and reporters.

Example Usage:
--------------
To run an evaluation from the command line:
  openevals run <suite_id_or_path> --target <target_system_name> --config <config_path>

To interact with the API (once server is running):
  POST /evaluate with EvaluationRequest payload.

Core Components:
- `GlobalConfig`, `EvalSuiteConfig`, `TargetSystemConfig`: Configuration models.
- `EvaluationRunner`: Orchestrates the evaluation process.
- `EvalTask`, `EvalCase`: Define the structure of evaluation tasks and test cases.
- Adapters (e.g., `OpenAIAdapter`, `HFAdapter`): Connect to AI systems.
- Graders (e.g., `exact_match`, `f1_metric`): Implement metric calculations.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import os
from pathlib import Path

# --- Version Information ---
try:
    # Attempt to get version from installed package metadata
    import importlib.metadata
    __version__ = importlib.metadata.version("openevals")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development environments or if not installed
    # This ensures __version__ is always defined.
    __version__ = "0.1.0-dev"

# --- Author Information ---
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "MIT" # Defined in pyproject.toml, can be stated here too

# --- Basic Logging Configuration ---
# Users can override this by configuring the root logger themselves,
# or by setting the OPENEVALS_LOG_LEVEL environment variable.
LOG_LEVEL_ENV = os.environ.get("OPENEVALS_LOG_LEVEL", "INFO").upper()
# Ensure LOG_LEVEL_ENV is a valid level string for logging.getLevelName
try:
    LOG_LEVEL = logging.getLevelName(LOG_LEVEL_ENV)
    if not isinstance(LOG_LEVEL, int): # Check if getLevelName returned a number
        LOG_LEVEL = logging.INFO # Default to INFO if invalid string
        logging.warning(
            f"Invalid OPENEVALS_LOG_LEVEL '{LOG_LEVEL_ENV}'. Defaulting to INFO."
        )
except ValueError:
    LOG_LEVEL = logging.INFO # Default to INFO if level name is not recognized
    logging.warning(
        f"Unknown OPENEVALS_LOG_LEVEL '{LOG_LEVEL_ENV}'. Defaulting to INFO."
    )


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__) # Logger for the OpenEvals package
logger.info(f"OpenEvals Framework Version: {__version__}, Effective Log Level: {logging.getLevelName(LOG_LEVEL)}")

# --- Expose Key Public Components ---
# This makes them available for direct import from the `OpenEvals` package.

# Configuration Models and Enums
from .config import (
    GlobalConfig,
    EvalSuiteConfig,
    TargetSystemConfig,
    EvalTaskConfigRef,
    StoredEvalType,
    EvalRunStatus,
    MetricType,
    InputFormat,
    OutputType,
    TransformationType,
    MetricResult,
    GradingOutput,
)

# Core Evaluation Definitions
from .core.definitions import (
    EvalInput,
    EvalOutputReference,
    EvalCase,
    EvalTask,
    EvalSuiteRun,
    MetricDefinition,
    GradingCriteria,
    TransformationStep,
    EvalTaskLoader,
)

# Core Evaluation Runner
from .core.runners import EvaluationRunner

# Core Adapters: Base class and registry functions
from .core.adapters import (
    AbstractAdapter,
    get_adapter,
    register_adapter,
    OpenAIAdapter, # Making common adapters directly importable
    HFAdapter,
    DummyAdapter,
)

# Core Graders: Protocol and registry functions
from .core.graders import (
    MetricFunction,
    get_grader,
    register_grader,
    exact_match, # Making common graders directly importable
    regex_match,
    semantic_similarity,
    f1_metric,
    llm_as_judge,
    apply_transformations, # Transformation utility
)

# Core Reporting Utilities
from .core.reporting import (
    generate_json_report,
    generate_console_report,
    generate_html_report,
    generate_detailed_html_report,
)

# Public API definition using __all__
# This controls what `from OpenEvals import *` imports.
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Config
    "GlobalConfig",
    "EvalSuiteConfig",
    "TargetSystemConfig",
    "EvalTaskConfigRef",
    "StoredEvalType",
    "EvalRunStatus",
    "MetricType",
    "InputFormat",
    "OutputType",
    "TransformationType",
    "MetricResult",
    "GradingOutput",
    # Definitions
    "EvalInput",
    "EvalOutputReference",
    "EvalCase",
    "EvalTask",
    "EvalSuiteRun",
    "MetricDefinition",
    "GradingCriteria",
    "TransformationStep",
    "EvalTaskLoader",
    # Runners
    "EvaluationRunner",
    # Adapters
    "AbstractAdapter",
    "get_adapter",
    "register_adapter",
    "OpenAIAdapter",
    "HFAdapter",
    "DummyAdapter",
    # Graders
    "MetricFunction",
    "get_grader",
    "register_grader",
    "exact_match",
    "regex_match",
    "semantic_similarity",
    "f1_metric",
    "llm_as_judge",
    "apply_transformations",
    # Reporting
    "generate_json_report",
    "generate_console_report",
    "generate_html_report",
    "generate_detailed_html_report",
]

logger.debug("OpenEvals framework public API exposed.") 