"""
Configuration loading and management for Open-Evals.

This module handles parsing YAML configuration files that define evaluation suites,
individual evaluation tasks, target systems, and other parameters.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
import os
import logging
from pydantic import BaseModel, validator, root_validator

from .data_structures import TargetSystemConfig, StoredEvalType
# EvalTaskConfig and EvalSuiteConfig will be more tightly coupled with core.definitions
# but we can have placeholders or forward declarations if needed, or define them more abstractly here.

logger = logging.getLogger(__name__)

@dataclass
class EvalTaskConfigRef:
    """Reference to an evaluation task definition."""
    task_id: str # Unique identifier for the task within the suite or globally
    # How to find this task definition: local file, registry, etc.
    source_type: StoredEvalType = StoredEvalType.LOCAL_FILE
    source_path: Optional[str] = None # Path to file if local, or name in registry
    # Optional: Override parameters for this specific instance of the task in a suite
    overrides: Dict[str, Any] = field(default_factory=dict)
    # Optional: Version of the task to use, if versioned
    version: Optional[str] = None 

@dataclass
class EvalSuiteConfig:
    """Configuration for a single evaluation suite."""
    suite_id: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    # List of tasks included in this suite, can be references or inline definitions (later)
    tasks: List[EvalTaskConfigRef] = field(default_factory=list)
    # Default target system for all tasks in this suite (can be overridden per task)
    default_target_system: Optional[str] = None # Name of a defined target system
    # Default reporting configuration
    default_report_formats: List[str] = field(default_factory=lambda: ["json", "console"])
    # Metadata for the suite
    metadata: Dict[str, Any] = field(default_factory=dict)

class GlobalConfig(BaseModel):
    """
    Global configuration for OpenEvals.
    """
    default_output_dir: str = "openevals_output"
    log_level: str = "INFO"
    evaluation_suites: Dict[str, Any] = {}
    target_systems: Dict[str, Any] = {}

    @validator('log_level')
    def validate_log_level(cls, v):
        """Ensure log_level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @root_validator(pre=True)
    def apply_env_overrides(cls, values):
        """Override config values with environment variables if present."""
        env_prefix = "OPENEVALS_"
        for field in cls.__fields__:
            env_var = f"{env_prefix}{field.upper()}"
            if env_var in os.environ:
                values[field] = os.environ[env_var]
        return values

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path], *additional_paths: Union[str, Path]) -> 'GlobalConfig':
        """
        Load configuration from one or more YAML files, merging them in order.
        Later files override earlier ones.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

        for path in additional_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Additional config file not found: {path}")
                continue
            with open(path, 'r') as f:
                additional_data = yaml.safe_load(f) or {}
                config_data.update(additional_data)

        return cls(**config_data)

    def get_evaluation_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """Get an evaluation suite by ID."""
        return self.evaluation_suites.get(suite_id)

    def get_target_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get a target system by ID."""
        return self.target_systems.get(system_id)

    def validate_paths(self):
        """Validate that all referenced paths exist."""
        for suite_id, suite in self.evaluation_suites.items():
            if "path" in suite:
                suite_path = Path(suite["path"])
                if not suite_path.exists():
                    logger.warning(f"Evaluation suite path not found for {suite_id}: {suite_path}")

        for system_id, system in self.target_systems.items():
            if "config_path" in system:
                system_path = Path(system["config_path"])
                if not system_path.exists():
                    logger.warning(f"Target system config path not found for {system_id}: {system_path}")

        output_dir = Path(self.default_output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

# Example usage (for testing or direct script runs):
if __name__ == "__main__": # pragma: no cover
    dummy_config_content = """
    default_output_dir: "./custom_eval_results"
    registry_path: "./eval_registry.yaml"
    metadata:
      project: "Archie Evals"
      version: "1.0"

    target_systems:
      - name: "archie_v1_dev"
        adapter_type: "custom_api"
        adapter_config:
          api_url: "http://localhost:8000/predict"
          api_key: "DEV_KEY_XYZ"
        version: "1.0-dev"
      - name: "openai_gpt4"
        adapter_type: "openai_chat"
        adapter_config:
          model_name: "gpt-4-turbo-preview"
          api_key_env_var: "OPENAI_API_KEY" # Example of referencing env var

    evaluation_suites:
      - suite_id: "core_engineering_skills"
        description: "Evaluates core engineering design and analysis skills."
        tags: ["engineering", "design", "core"]
        default_target_system: "archie_v1_dev"
        tasks:
          - task_id: "beam_deflection_analysis"
            source_type: "local_file"
            source_path: "./eval_tasks/beam_deflection.yaml"
            overrides:
              difficulty: "medium"
          - task_id: "gear_ratio_calculation"
            source_type: "registry_reference"
            source_path: "standard_engineering_calculations/gear_ratio_v1"
      - suite_id: "general_qa"
        description: "General question answering capabilities."
        tags: ["qa", "general"]
        default_target_system: "openai_gpt4"
        tasks:
          - task_id: "common_knowledge_test"
            source_type: "local_file"
            source_path: "./eval_tasks/common_knowledge.yaml"
    """
    test_yaml_path = Path("./temp_test_global_config.yaml")
    with open(test_yaml_path, 'w') as f:
        f.write(dummy_config_content)
    
    print(f"Attempting to load config from: {test_yaml_path.resolve()}")
    try:
        global_config = GlobalConfig.from_yaml(test_yaml_path)
        print("Successfully loaded GlobalConfig:")
        print(f"  Default Output Dir: {global_config.default_output_dir}")
        print(f"  Target Systems: {len(global_config.target_systems)}")
        if global_config.target_systems:
            print(f"    Example Target System: {global_config.target_systems[0].name}")
        print(f"  Evaluation Suites: {len(global_config.evaluation_suites)}")
        if global_config.evaluation_suites:
            print(f"    Example Eval Suite: {global_config.evaluation_suites[0].suite_id}")
            if global_config.evaluation_suites[0].tasks:
                print(f"      Example Task Ref: {global_config.evaluation_suites[0].tasks[0].task_id}")
        
        # Test serialization
        dict_output = global_config.to_dict()
        print("\nSerialized to dict:", dict_output["evaluation_suites"][0]["tasks"][0])
        
        yaml_save_path = Path("./temp_test_global_config_saved.yaml")
        global_config.to_yaml(yaml_save_path)
        print(f"\nConfiguration saved to: {yaml_save_path.resolve()}")

    except Exception as e:
        print(f"Error during example config processing: {e}")
    finally:
        # Clean up temp files
        if test_yaml_path.exists():
            test_yaml_path.unlink()
        if yaml_save_path.exists():
            yaml_save_path.unlink() 