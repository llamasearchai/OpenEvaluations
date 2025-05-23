"""
OpenEvaluations: AI Evaluation Framework
========================================

OpenEvaluations is a comprehensive, modular framework designed for robustly evaluating
AI systems across various tasks and metrics, with specialized focus on scientific domains.
It supports configurable evaluation suites, diverse AI system adapters, a rich set of 
grading metrics, and flexible reporting capabilities.

Key Features:
- Define evaluation tasks and suites using YAML configuration.
- Interface with different AI models (OpenAI, Hugging Face, etc.) via adapters.
- Apply a wide range of metrics (exact match, F1, ROUGE, semantic similarity, LLM-as-judge).
- Generate detailed reports in JSON, console, and HTML formats.
- Run evaluations via a Command Line Interface (CLI) or a FastAPI-based API.
- Extensible design for adding custom adapters, metrics, and reporters.
- Comprehensive scientific evaluation modules for biology, genomics, virology, neuroscience.

Example Usage:
--------------
To run an evaluation from the command line:
  openevaluations run <suite_id_or_path> --target <target_system_name> --config <config_path>

To interact with the API (once server is running):
  POST /evaluate with EvaluationRequest payload.

Core Components:
- `GlobalConfig`, `EvalSuiteConfig`, `TargetSystemConfig`: Configuration models.
- `EvaluationRunner`: Orchestrates the evaluation process.
- `EvalTask`, `EvalCase`: Define the structure of evaluation tasks and test cases.
- Adapters (e.g., `OpenAIAdapter`, `HFAdapter`): Connect to AI systems.
- Graders (e.g., `exact_match`, `f1_metric`): Implement metric calculations.
- Scientific Modules: Biology, Genomics, Virology, Neuroscience evaluation suites.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import os
from pathlib import Path

# --- Version Information ---
try:
    # Attempt to get version from installed package metadata
    import importlib.metadata
    __version__ = importlib.metadata.version("openevaluations")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development environments or if not installed
    # This ensures __version__ is always defined.
    __version__ = "2.0.0"

# --- Author Information ---
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "MIT"
__homepage__ = "https://github.com/llamasearchai/OpenEvaluations"

# --- Basic Logging Configuration ---
# Users can override this by configuring the root logger themselves,
# or by setting the OPENEVALUATIONS_LOG_LEVEL environment variable.
LOG_LEVEL_ENV = os.environ.get("OPENEVALUATIONS_LOG_LEVEL", "INFO").upper()
# Ensure LOG_LEVEL_ENV is a valid level string for logging.getLevelName
try:
    LOG_LEVEL = logging.getLevelName(LOG_LEVEL_ENV)
    if not isinstance(LOG_LEVEL, int): # Check if getLevelName returned a number
        LOG_LEVEL = logging.INFO # Default to INFO if invalid string
        logging.warning(
            f"Invalid OPENEVALUATIONS_LOG_LEVEL '{LOG_LEVEL_ENV}'. Defaulting to INFO."
        )
except ValueError:
    LOG_LEVEL = logging.INFO # Default to INFO if level name is not recognized
    logging.warning(
        f"Unknown OPENEVALUATIONS_LOG_LEVEL '{LOG_LEVEL_ENV}'. Defaulting to INFO."
    )


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__) # Logger for the OpenEvaluations package
logger.info(f"OpenEvaluations Framework Version: {__version__}, Effective Log Level: {logging.getLevelName(LOG_LEVEL)}")

# --- Expose Key Public Components ---
# This makes them available for direct import from the `openevaluations` package.

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

# Scientific Evaluation Modules
try:
    from .evals.genomics import (
        GenomicsEvaluationSuite,
        GenomicVariant,
        GeneExpressionData,
        PhylogeneticTree,
        create_genomics_evaluation_suite,
        genomics_variant_accuracy,
        pathway_identification_accuracy,
        phylogenetic_interpretation_accuracy
    )
except ImportError:
    logger.warning("Genomics evaluation module not available")

try:
    from .evals.virology import (
        VirologyEvaluationSuite,
        ViralGenome,
        ViralProtein,
        EpidemiologicalData,
        VaccineCandidate,
        create_virology_evaluation_suite,
        viral_structure_accuracy,
        pathogenesis_mechanism_accuracy,
        epidemiological_calculation_accuracy
    )
except ImportError:
    logger.warning("Virology evaluation module not available")

try:
    from .evals.neuroscience import (
        NeuroscienceEvaluationSuite,
        NeuralCircuit,
        NeuroimagingData,
        CognitiveTask,
        NeurologicalDisorder,
        create_neuroscience_evaluation_suite,
        neural_circuit_accuracy,
        cognitive_profile_accuracy
    )
except ImportError:
    logger.warning("Neuroscience evaluation module not available")

# Public API definition using __all__
# This controls what `from openevaluations import *` imports.
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__homepage__",
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
    # Scientific Evaluation Suites
    "GenomicsEvaluationSuite",
    "VirologyEvaluationSuite", 
    "NeuroscienceEvaluationSuite",
    # Scientific Data Structures
    "GenomicVariant",
    "GeneExpressionData",
    "PhylogeneticTree",
    "ViralGenome",
    "ViralProtein",
    "EpidemiologicalData",
    "VaccineCandidate",
    "NeuralCircuit",
    "NeuroimagingData",
    "CognitiveTask",
    "NeurologicalDisorder",
    # Scientific Suite Creators
    "create_genomics_evaluation_suite",
    "create_virology_evaluation_suite",
    "create_neuroscience_evaluation_suite",
    # Scientific Graders
    "genomics_variant_accuracy",
    "pathway_identification_accuracy",
    "phylogenetic_interpretation_accuracy",
    "viral_structure_accuracy",
    "pathogenesis_mechanism_accuracy",
    "epidemiological_calculation_accuracy",
    "neural_circuit_accuracy",
    "cognitive_profile_accuracy",
]

logger.debug("OpenEvaluations framework public API exposed.")
logger.info(f"Scientific evaluation modules loaded: Genomics, Virology, Neuroscience") 