# Registry for discovering and managing evaluations 

"""
Evaluation task registry implementation
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import asdict

from openevals.core.definitions import EvalTask
from openevals.config.data_structures import StoredEvalType

logger = logging.getLogger(__name__)

class EvaluationRegistry:
    """Central registry for evaluation tasks and suites"""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path) if registry_path else None
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.suites: Dict[str, Dict[str, Any]] = {}
        
        if self.registry_path and self.registry_path.exists():
            self.load_registry()
    
    def load_registry(self) -> None:
        """Load registry from YAML file"""
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = yaml.safe_load(f) or {}
            
            self.tasks = registry_data.get('tasks', {})
            self.suites = registry_data.get('suites', {})
            logger.info(f"Loaded registry from {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to load registry: {str(e)}")
            raise
    
    def save_registry(self) -> None:
        """Save registry to YAML file"""
        if not self.registry_path:
            raise ValueError("No registry path configured")
            
        try:
            registry_data = {
                'tasks': self.tasks,
                'suites': self.suites
            }
            
            with open(self.registry_path, 'w') as f:
                yaml.dump(registry_data, f, default_flow_style=False)
            logger.info(f"Saved registry to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save registry: {str(e)}")
            raise
    
    def register_task(self, task: EvalTask, version: str = "1.0", 
                     description: Optional[str] = None) -> None:
        """Register a new evaluation task"""
        if not task.task_id:
            raise ValueError("Task must have an ID")
            
        task_data = {
            'definition': asdict(task),
            'version': version,
            'description': description or task.description
        }
        
        self.tasks[task.task_id] = task_data
        logger.info(f"Registered task: {task.task_id} (v{version})")
    
    def get_task(self, task_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a task definition from the registry"""
        if task_id not in self.tasks:
            return None
            
        task_data = self.tasks[task_id]
        if version and task_data.get('version') != version:
            return None
            
        return task_data
    
    def register_suite(self, suite_id: str, task_ids: List[str], 
                      version: str = "1.0", description: Optional[str] = None) -> None:
        """Register a new evaluation suite"""
        if not suite_id:
            raise ValueError("Suite must have an ID")
        if not task_ids:
            raise ValueError("Suite must contain at least one task")
            
        suite_data = {
            'task_ids': task_ids,
            'version': version,
            'description': description
        }
        
        self.suites[suite_id] = suite_data
        logger.info(f"Registered suite: {suite_id} (v{version})")
    
    def get_suite(self, suite_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a suite definition from the registry"""
        if suite_id not in self.suites:
            return None
            
        suite_data = self.suites[suite_id]
        if version and suite_data.get('version') != version:
            return None
            
        return suite_data 