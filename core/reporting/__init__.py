"""
Evaluation reporting utilities
"""
import json
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
from dataclasses import asdict

from core.definitions import EvalSuiteRun, EvalTask
logger = logging.getLogger(__name__)

def generate_json_report(suite_run: EvalSuiteRun, output_dir: str) -> str:
    """Generate a JSON report for an evaluation suite run"""
    report_path = Path(output_dir) / f"report_{suite_run.run_id}.json"
    report_data = {
        "metadata": {
            "run_id": suite_run.run_id,
            "start_time": suite_run.start_time,
            "end_time": suite_run.end_time,
            "status": suite_run.status,
            "target_system": suite_run.target_system_config.name
        },
        "results": [asdict(task) for task in suite_run.tasks]
    }
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"JSON report saved to: {report_path}")
        return str(report_path)
    except Exception as e:
        logger.error(f"Failed to generate JSON report: {str(e)}")
        raise

def generate_console_report(suite_run: EvalSuiteRun) -> None:
    """Print evaluation results to console"""
    print(f"\nEvaluation Report - Run ID: {suite_run.run_id}")
    print(f"Target System: {suite_run.target_system_config.name}")
    print(f"Status: {suite_run.status}")
    print(f"Start: {suite_run.start_time}")
    print(f"End: {suite_run.end_time}")
    print("\nTask Results:")
    
    for task in suite_run.tasks:
        print(f"\nTask: {task.task_id}")
        print(f"  Status: {task.status}")
        print(f"  Overall Score: {task.overall_score or 'N/A'}")
        
        if task.summary_metrics:
            print("  Summary Metrics:")
            for name, value in task.summary_metrics.items():
                print(f"    {name}: {value}")

def generate_html_report(suite_run: EvalSuiteRun, output_dir: str) -> str:
    """Generate an HTML report for an evaluation suite run"""
    report_path = Path(output_dir) / f"report_{suite_run.run_id}.html"
    
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenEvals Report - {suite_run.run_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .task {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
                .metrics {{ margin-left: 20px; }}
            </style>
        </head>
        <body>
            <h1>Evaluation Report</h1>
            <p><strong>Run ID:</strong> {suite_run.run_id}</p>
            <p><strong>Target System:</strong> {suite_run.target_system_config.name}</p>
            <p><strong>Status:</strong> {suite_run.status}</p>
            <p><strong>Start Time:</strong> {suite_run.start_time}</p>
            <p><strong>End Time:</strong> {suite_run.end_time}</p>
            
            <h2>Tasks</h2>
            {_generate_task_html(suite_run.tasks)}
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        logger.info(f"HTML report saved to: {report_path}")
        return str(report_path)
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {str(e)}")
        raise

def _generate_task_html(tasks: List[EvalTask]) -> str:
    """Generate HTML for task results"""
    task_html = []
    for task in tasks:
        metrics_html = "".join(
            f"<li><strong>{name}:</strong> {value}</li>"
            for name, value in (task.summary_metrics or {}).items()
        )
        
        task_html.append(f"""
        <div class="task">
            <h3>{task.task_id}</h3>
            <p><strong>Status:</strong> {task.status}</p>
            <p><strong>Overall Score:</strong> {task.overall_score or 'N/A'}</p>
            <ul class="metrics">{metrics_html}</ul>
        </div>
        """)
    
    return "\n".join(task_html) 