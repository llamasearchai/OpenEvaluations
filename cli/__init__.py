"""
OpenEvals CLI Package
====================

Command-line interface for the OpenEvals framework.
Provides commands for running evaluations, managing configurations, and more.
"""

from .main import app, run_evaluation_command, list_suites_command, validate_config_command

__all__ = [
    "app",
    "run_evaluation_command", 
    "list_suites_command",
    "validate_config_command"
] 