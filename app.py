"""
OpenEvals FastAPI Server
========================

Comprehensive web server for the OpenEvals AI evaluation framework.
Provides both web interface and REST API endpoints for running evaluations.

Features:
- Web interface with real-time updates
- REST API for programmatic access
- WebSocket support for live progress tracking
- Background task processing for evaluations
- File upload and management
- Result export and sharing
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
import uvicorn
import yaml

# Import OpenEvals components
from config import GlobalConfig, EvalSuiteConfig, TargetSystemConfig
from core.runners import EvaluationRunner
from core.definitions import EvalTaskLoader
from core.adapters import get_adapter
from core.reporting import generate_json_report, generate_html_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenEvals",
    description="Comprehensive AI Evaluation Framework",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Security
security = HTTPBasic()

# In-memory storage for demonstration (use database in production)
evaluation_runs: Dict[str, Dict] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# --- Pydantic Models ---

class EvaluationRequest(BaseModel):
    """Request model for starting a new evaluation"""
    suite_id: str = Field(..., description="Evaluation suite identifier")
    target_system: str = Field(..., description="Target system configuration")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    name: Optional[str] = Field(None, description="Optional evaluation name")
    description: Optional[str] = Field(None, description="Optional description")

class EvaluationStatus(BaseModel):
    """Response model for evaluation status"""
    run_id: str
    status: str
    progress: float
    current_task: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    results: Optional[Dict[str, Any]]
    error: Optional[str]

class SystemStatus(BaseModel):
    """System health status model"""
    status: str
    version: str
    uptime: float
    active_evaluations: int
    total_evaluations: int
    memory_usage: Optional[float]

# --- Helper Functions ---

async def notify_websocket_clients(run_id: str, data: Dict[str, Any]) -> None:
    """Send real-time updates to WebSocket clients"""
    if run_id in websocket_connections:
        disconnected = []
        for websocket in websocket_connections[run_id]:
            try:
                await websocket.send_json(data)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            websocket_connections[run_id].remove(ws)

def create_evaluation_run(request: EvaluationRequest) -> str:
    """Create a new evaluation run entry"""
    run_id = str(uuid.uuid4())
    
    evaluation_runs[run_id] = {
        "run_id": run_id,
        "status": "pending",
        "progress": 0.0,
        "current_task": None,
        "start_time": datetime.utcnow(),
        "end_time": None,
        "results": None,
        "error": None,
        "config": {
            "suite_id": request.suite_id,
            "target_system": request.target_system,
            **request.config
        },
        "name": request.name,
        "description": request.description
    }
    
    return run_id

async def run_evaluation_background(run_id: str, request: EvaluationRequest) -> None:
    """Background task to run evaluation"""
    try:
        # Update status to running
        evaluation_runs[run_id]["status"] = "running"
        evaluation_runs[run_id]["current_task"] = "Initializing evaluation..."
        
        await notify_websocket_clients(run_id, {
            "type": "status_update",
            "run_id": run_id,
            "status": "running",
            "progress": 5.0,
            "current_task": "Initializing evaluation..."
        })
        
        # Load configuration
        config = GlobalConfig()
        
        # Find evaluation suite
        suite_config = None
        for suite in config.evaluation_suites:
            if suite.id == request.suite_id:
                suite_config = suite
                break
        
        if not suite_config:
            raise ValueError(f"Evaluation suite '{request.suite_id}' not found")
        
        # Find target system
        target_config = None
        for target in config.target_systems:
            if target.name == request.target_system:
                target_config = target
                break
        
        if not target_config:
            raise ValueError(f"Target system '{request.target_system}' not found")
        
        # Update progress
        evaluation_runs[run_id]["progress"] = 10.0
        evaluation_runs[run_id]["current_task"] = "Loading tasks..."
        
        await notify_websocket_clients(run_id, {
            "type": "status_update",
            "run_id": run_id,
            "progress": 10.0,
            "current_task": "Loading tasks..."
        })
        
        # Load tasks
        task_loader = EvalTaskLoader(config)
        tasks = []
        for task_ref in suite_config.tasks:
            task = task_loader.load_task(task_ref.task_id)
            tasks.append(task)
        
        # Initialize adapter
        adapter = get_adapter(target_config.adapter_type)
        adapter.configure(target_config.config)
        
        # Update progress
        evaluation_runs[run_id]["progress"] = 20.0
        evaluation_runs[run_id]["current_task"] = "Starting evaluation runner..."
        
        await notify_websocket_clients(run_id, {
            "type": "status_update",
            "run_id": run_id,
            "progress": 20.0,
            "current_task": "Starting evaluation runner..."
        })
        
        # Run evaluation
        runner = EvaluationRunner(config)
        
        # Mock evaluation with progress updates
        total_tasks = len(tasks)
        for i, task in enumerate(tasks):
            progress = 20.0 + (i / total_tasks) * 70.0  # 20% to 90%
            
            evaluation_runs[run_id]["progress"] = progress
            evaluation_runs[run_id]["current_task"] = f"Processing task {i+1}/{total_tasks}: {task.name}"
            
            await notify_websocket_clients(run_id, {
                "type": "task_progress",
                "run_id": run_id,
                "progress": progress,
                "current_task": evaluation_runs[run_id]["current_task"],
                "task_index": i,
                "total_tasks": total_tasks
            })
            
            # Simulate processing time
            await asyncio.sleep(1.0)
        
        # Generate mock results
        results = {
            "total_tasks": len(tasks),
            "passed_tasks": len(tasks) - 2,
            "failed_tasks": 2,
            "overall_score": 0.87,
            "execution_time": 120.5,
            "task_results": [
                {
                    "task_name": f"Task {i+1}",
                    "score": 0.85 + (i * 0.02),
                    "status": "passed" if i < len(tasks) - 2 else "failed",
                    "execution_time": 2.5 + (i * 0.1)
                }
                for i in range(len(tasks))
            ]
        }
        
        # Update final status
        evaluation_runs[run_id]["status"] = "completed"
        evaluation_runs[run_id]["progress"] = 100.0
        evaluation_runs[run_id]["current_task"] = "Evaluation completed"
        evaluation_runs[run_id]["end_time"] = datetime.utcnow()
        evaluation_runs[run_id]["results"] = results
        
        await notify_websocket_clients(run_id, {
            "type": "evaluation_complete",
            "run_id": run_id,
            "status": "completed",
            "progress": 100.0,
            "results": results
        })
        
        logger.info(f"Evaluation {run_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation {run_id} failed: {str(e)}")
        
        evaluation_runs[run_id]["status"] = "failed"
        evaluation_runs[run_id]["error"] = str(e)
        evaluation_runs[run_id]["end_time"] = datetime.utcnow()
        
        await notify_websocket_clients(run_id, {
            "type": "evaluation_failed",
            "run_id": run_id,
            "status": "failed",
            "error": str(e)
        })

# --- Web Interface Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page"""
    # Get recent runs
    recent_runs = list(evaluation_runs.values())
    recent_runs.sort(key=lambda x: x["start_time"], reverse=True)
    recent_runs = recent_runs[:10]  # Last 10 runs
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "recent_runs": recent_runs
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Comprehensive dashboard page"""
    runs = list(evaluation_runs.values())
    runs.sort(key=lambda x: x["start_time"], reverse=True)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "runs": runs
    })

@app.get("/new-evaluation", response_class=HTMLResponse)
async def new_evaluation(request: Request):
    """New evaluation creation page"""
    return templates.TemplateResponse("new_evaluation.html", {
        "request": request
    })

@app.get("/evaluation/{run_id}", response_class=HTMLResponse)
async def evaluation_detail(request: Request, run_id: str):
    """Detailed evaluation view"""
    if run_id not in evaluation_runs:
        return templates.TemplateResponse("errors/404.html", {
            "request": request
        }, status_code=404)
    
    run_info = evaluation_runs[run_id]
    
    return templates.TemplateResponse("evaluation_detail.html", {
        "request": request,
        "run_info": run_info
    })

# --- API Routes ---

@app.post("/api/start-evaluation")
async def start_evaluation(
    background_tasks: BackgroundTasks,
    suite_id: str = Form(...),
    target_system: str = Form(...),
    evaluation_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    max_workers: int = Form(4),
    timeout: int = Form(30),
    config_file: Optional[UploadFile] = File(None)
):
    """Start a new evaluation"""
    try:
        # Process uploaded config file if provided
        custom_config = {}
        if config_file and config_file.filename:
            if not config_file.filename.endswith(('.yaml', '.yml')):
                raise HTTPException(400, "Config file must be YAML format")
            
            content = await config_file.read()
            try:
                custom_config = yaml.safe_load(content.decode('utf-8'))
            except yaml.YAMLError as e:
                raise HTTPException(400, f"Invalid YAML file: {str(e)}")
        
        # Create evaluation request
        request_data = EvaluationRequest(
            suite_id=suite_id,
            target_system=target_system,
            name=evaluation_name,
            description=description,
            config={
                "max_workers": max_workers,
                "timeout": timeout,
                **custom_config
            }
        )
        
        # Create evaluation run
        run_id = create_evaluation_run(request_data)
        
        # Start background task
        background_tasks.add_task(run_evaluation_background, run_id, request_data)
        
        return JSONResponse({
            "success": True,
            "run_id": run_id,
            "message": "Evaluation started successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to start evaluation: {str(e)}")
        raise HTTPException(500, f"Failed to start evaluation: {str(e)}")

@app.get("/api/evaluation/{run_id}/status")
async def get_evaluation_status(run_id: str) -> EvaluationStatus:
    """Get evaluation status"""
    if run_id not in evaluation_runs:
        raise HTTPException(404, "Evaluation run not found")
    
    run_data = evaluation_runs[run_id]
    
    return EvaluationStatus(
        run_id=run_data["run_id"],
        status=run_data["status"],
        progress=run_data["progress"],
        current_task=run_data["current_task"],
        start_time=run_data["start_time"],
        end_time=run_data["end_time"],
        results=run_data["results"],
        error=run_data["error"]
    )

@app.get("/api/evaluation/{run_id}/results")
async def get_evaluation_results(run_id: str):
    """Get detailed evaluation results"""
    if run_id not in evaluation_runs:
        raise HTTPException(404, "Evaluation run not found")
    
    run_data = evaluation_runs[run_id]
    
    if run_data["status"] != "completed":
        raise HTTPException(400, "Evaluation not completed yet")
    
    if not run_data["results"]:
        raise HTTPException(404, "Results not available")
    
    return JSONResponse(run_data["results"])

@app.get("/api/evaluations")
async def list_evaluations(limit: int = 50, offset: int = 0):
    """List all evaluations with pagination"""
    runs = list(evaluation_runs.values())
    runs.sort(key=lambda x: x["start_time"], reverse=True)
    
    total = len(runs)
    paginated_runs = runs[offset:offset + limit]
    
    return JSONResponse({
        "evaluations": paginated_runs,
        "total": total,
        "limit": limit,
        "offset": offset
    })

@app.get("/api/system/status")
async def get_system_status() -> SystemStatus:
    """Get system health status"""
    import time
    
    active_evaluations = len([r for r in evaluation_runs.values() if r["status"] == "running"])
    total_evaluations = len(evaluation_runs)
    
    try:
        import psutil
        memory_usage = psutil.virtual_memory().percent
    except:
        memory_usage = None
    
    return SystemStatus(
        status="healthy",
        version="1.0.0",
        uptime=time.time(),
        active_evaluations=active_evaluations,
        total_evaluations=total_evaluations,
        memory_usage=memory_usage
    )

# --- WebSocket Routes ---

@app.websocket("/ws/evaluation/{run_id}")
async def websocket_evaluation_updates(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time evaluation updates"""
    await websocket.accept()
    
    # Add to connections
    if run_id not in websocket_connections:
        websocket_connections[run_id] = []
    websocket_connections[run_id].append(websocket)
    
    try:
        # Send current status
        if run_id in evaluation_runs:
            await websocket.send_json({
                "type": "status_update",
                "run_id": run_id,
                **evaluation_runs[run_id]
            })
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        # Remove from connections
        if run_id in websocket_connections:
            websocket_connections[run_id].remove(websocket)
            if not websocket_connections[run_id]:
                del websocket_connections[run_id]

# --- Error Handlers ---

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 error handler"""
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "message": "The requested resource was not found"}
        )
    
    return templates.TemplateResponse("errors/404.html", {
        "request": request
    }, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Custom 500 error handler"""
    logger.error(f"Internal server error: {str(exc)}")
    
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error", 
                "message": "An unexpected error occurred"
            }
        )
    
    return templates.TemplateResponse("errors/500.html", {
        "request": request,
        "error_description": str(exc),
        "error_id": str(uuid.uuid4())[:8]
    }, status_code=500)

# --- Startup Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting OpenEvals server...")
    
    # Create necessary directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("OpenEvals server started successfully")

# --- Main Entry Point ---

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 