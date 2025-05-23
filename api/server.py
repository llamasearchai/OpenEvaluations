"""
FastAPI server for OpenEvals API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

from openevals import EvaluationRunner, GlobalConfig
from openevals.core.definitions import EvalSuiteRun
from openevals.config import EvalSuiteConfig, TargetSystemConfig

app = FastAPI(
    title="OpenEvals API",
    description="REST API for running and managing AI evaluations",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

class EvaluationRequest(BaseModel):
    suite_id: str
    target_system: str
    config_path: str = "config.yaml"
    output_dir: Optional[str] = None
    mode: str = "parallel"
    max_workers: int = 4

class EvaluationResponse(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    results_path: Optional[str] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize global resources"""
    logger.info("Starting OpenEvals API server")

@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Execute an evaluation suite via API"""
    try:
        # Load configuration
        global_config_path = Path(request.config_path)
        if not global_config_path.exists():
            logger.error(f"Global config file not found at: {global_config_path}")
            raise HTTPException(status_code=500, detail=f"Global config file not found: {global_config_path}")
        global_config = GlobalConfig.from_yaml(str(global_config_path))
        
        # Find the requested suite and target system
        suite = global_config.get_evaluation_suite(request.suite_id)
        if not suite:
            raise HTTPException(status_code=404, detail=f"Suite not found: {request.suite_id}")
            
        target_system = global_config.get_target_system(request.target_system)
        if not target_system:
            raise HTTPException(status_code=404, detail=f"Target system not found: {request.target_system}")
        
        # Initialize runner
        runner = EvaluationRunner(
            global_config=global_config,
            max_workers=request.max_workers
        )
        
        # Run evaluation
        suite_run = runner.run_suite(
            suite_config=suite,
            target_system=target_system,
            output_dir=Path(request.output_dir) if request.output_dir else None
        )
        
        # Save the EvalSuiteRun object for later retrieval
        if suite_run.output_dir:
            results_file_path = Path(suite_run.output_dir) / f"run_{suite_run.run_id}_results.json"
            with open(results_file_path, "w") as f:
                json.dump(suite_run.model_dump(), f, indent=4)
            logger.info(f"Evaluation results saved to {results_file_path}")
            results_path_str = str(results_file_path)
        else:
            results_path_str = None
            logger.warning(f"Output directory not specified for run {suite_run.run_id}, results not saved to file.")

        return EvaluationResponse(
            run_id=suite_run.run_id,
            status=str(suite_run.status.value),
            start_time=suite_run.start_time.isoformat(),
            end_time=suite_run.end_time.isoformat() if suite_run.end_time else None,
            results_path=results_path_str
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"API evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def load_and_reconstruct_eval_suite_run_from_json(run_id, global_config):
    """Helper to load EvalSuiteRun from its JSON file."""
    # This assumes a consistent output directory structure or a database lookup.
    # For now, let's search common output locations or require a config setting.
    # This is a simplified example; a real system might use a database or a configured base path.
    
    # Attempt to find the run output directory. This is illustrative.
    # A robust solution would need a reliable way to map run_id to its output path.
    # For now, we'll assume a 'runs' directory in the current working directory or configured path.
    
    # Option 1: Check a configured base path (e.g., from GlobalConfig or environment)
    # For this example, let's assume a `default_output_dir` in GlobalConfig if it exists
    base_output_dir = Path(getattr(global_config, 'default_output_dir', 'openevals_output'))

    # Try to find a directory that might contain this run_id
    # This is a very naive search. A better way is to store the exact path when the run is created.
    possible_run_dir = base_output_dir / run_id # if run_id is the directory name
    results_file = possible_run_dir / f"run_{run_id}_results.json"

    if not results_file.exists():
        # Fallback: Search for any directory in base_output_dir that might contain the file
        # This is inefficient and not recommended for production.
        logger.warning(f"Results file not found at {results_file}. Searching...")
        found = False
        for item in base_output_dir.iterdir():
            if item.is_dir():
                potential_file = item / f"run_{run_id}_results.json"
                if potential_file.exists():
                    results_file = potential_file
                    found = True
                    break
        if not found:
            logger.error(f"Could not find results file for run_id: {run_id} in {base_output_dir} or its subdirectories.")
            return None
            
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
        # Re-construct the EvalSuiteRun object. This assumes EvalSuiteRun can be Pydantic-parsed.
        return EvalSuiteRun(**data)
    except FileNotFoundError:
        logger.error(f"Results file not found for run_id: {run_id} at {results_file}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON for run_id: {run_id} from {results_file}")
        return None
    except Exception as e:
        logger.error(f"Failed to load run {run_id}: {e}")
        return None

@app.get("/runs/{run_id}", response_model=EvalSuiteRun)
async def get_run_results(run_id: str, config_path: str = "config.yaml"):
    """Get results for a specific evaluation run"""
    try:
        global_config_path = Path(config_path)
        if not global_config_path.exists():
            raise HTTPException(status_code=500, detail=f"Global config file not found: {config_path}")
        global_config = GlobalConfig.from_yaml(str(global_config_path))

        suite_run = load_and_reconstruct_eval_suite_run_from_json(run_id, global_config)
        if suite_run:
            return suite_run
        else:
            raise HTTPException(status_code=404, detail=f"Results for run_id '{run_id}' not found.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving results for run_id {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 