"""
Evaluation runner implementation
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
from datetime import datetime
import logging

from openevals.core.definitions import (
    EvalTask,
    EvalSuiteRun,
    EvalTaskLoader
)
from openevals.config import (
    GlobalConfig,
    EvalSuiteConfig,
    TargetSystemConfig
)

logger = logging.getLogger(__name__)

class EvaluationRunner:
    """Orchestrates the execution of evaluation suites"""
    
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.task_loader = EvalTaskLoader(global_config.registry_path)
        
    def run_suite(
        self,
        suite_config: EvalSuiteConfig,
        target_system: TargetSystemConfig,
        output_dir: Optional[str] = None
    ) -> EvalSuiteRun:
        """Execute a full evaluation suite"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not output_dir:
            output_dir = self.global_config.default_output_dir
            
        suite_run = EvalSuiteRun(
            suite_config=suite_config,
            tasks=[],
            target_system_config=target_system,
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            output_dir=output_dir
        )
        
        try:
            # Load all tasks
            tasks = []
            for task_ref in suite_config.tasks:
                task = self.task_loader.load_task(task_ref)
                tasks.append(task)
                
            suite_run.tasks = tasks
            
            # TODO: Implement actual task execution
            # This would involve:
            # 1. Initializing the adapter
            # 2. Running each case
            # 3. Applying metrics
            # 4. Aggregating results
            
            suite_run.status = "completed"
            suite_run.end_time = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Suite execution failed: {str(e)}")
            suite_run.status = "failed"
            raise
            
        return suite_run 