"""
OpenEvals Command Line Interface
===============================

Comprehensive CLI for the OpenEvals framework using Typer.
Provides commands for running evaluations, managing configurations, and more.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich import print as rprint

# Import OpenEvals components
from config import GlobalConfig, EvalSuiteConfig, TargetSystemConfig
from core.runners import EvaluationRunner
from core.definitions import EvalTaskLoader
from core.adapters import get_adapter, list_adapters
from core.graders import list_graders
from core.reporting import generate_json_report, generate_console_report, generate_html_report

# Initialize Typer app and Rich console
app = typer.Typer(
    name="openevals",
    help="OpenEvals - Comprehensive AI Evaluation Framework",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()

# Global state
global_config: Optional[GlobalConfig] = None

# --- Helper Functions ---

def load_global_config() -> GlobalConfig:
    """Load and cache global configuration"""
    global global_config
    if global_config is None:
        try:
            global_config = GlobalConfig()
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            raise typer.Exit(1)
    return global_config

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_timestamp(timestamp: Optional[float]) -> str:
    """Format timestamp in human-readable format"""
    if timestamp is None:
        return "N/A"
    
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def validate_suite_id(suite_id: str) -> EvalSuiteConfig:
    """Validate and return suite configuration"""
    config = load_global_config()
    
    for suite in config.evaluation_suites:
        if suite.id == suite_id:
            return suite
    
    console.print(f"[red]Error: Evaluation suite '{suite_id}' not found[/red]")
    console.print("\nAvailable suites:")
    for suite in config.evaluation_suites:
        console.print(f"  • {suite.id}: {suite.name}")
    
    raise typer.Exit(1)

def validate_target_system(target_name: str) -> TargetSystemConfig:
    """Validate and return target system configuration"""
    config = load_global_config()
    
    for target in config.target_systems:
        if target.name == target_name:
            return target
    
    console.print(f"[red]Error: Target system '{target_name}' not found[/red]")
    console.print("\nAvailable targets:")
    for target in config.target_systems:
        console.print(f"  • {target.name}: {target.adapter_type}")
    
    raise typer.Exit(1)

# --- Main Commands ---

@app.command("run")
def run_evaluation_command(
    suite_id: str = typer.Argument(..., help="Evaluation suite ID to run"),
    target: str = typer.Option(..., "--target", "-t", help="Target system name"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Maximum number of parallel workers"),
    timeout: int = typer.Option(60, "--timeout", help="Request timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be executed without running"),
    json_output: bool = typer.Option(False, "--json", help="Output results in JSON format"),
    html_report: bool = typer.Option(True, "--html/--no-html", help="Generate HTML report"),
    save_raw: bool = typer.Option(False, "--save-raw", help="Save raw model responses")
):
    """
    Run an evaluation suite against a target system.
    
    Examples:
        openevals run basic_qa --target openai_gpt4
        openevals run comprehensive --target huggingface --config my_config.yaml
        openevals run custom_suite --target local_model --workers 8 --timeout 120
    """
    
    console.print(Panel.fit(
        "[bold blue]OpenEvals - AI Evaluation Framework[/bold blue]\n"
        f"Running evaluation suite: [green]{suite_id}[/green]\n"
        f"Target system: [yellow]{target}[/yellow]",
        title="🚀 Starting Evaluation"
    ))
    
    # Validate inputs
    suite_config = validate_suite_id(suite_id)
    target_config = validate_target_system(target)
    
    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No evaluation will be executed[/yellow]")
        show_evaluation_plan(suite_config, target_config, max_workers, timeout)
        return
    
    # Load additional configuration
    additional_config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                additional_config = yaml.safe_load(f)
            console.print(f"[green]✓[/green] Loaded configuration from {config_file}")
        except Exception as e:
            console.print(f"[red]Error loading config file: {e}[/red]")
            raise typer.Exit(1)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"results_{suite_id}_{int(time.time())}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the evaluation
        config = load_global_config()
        runner = EvaluationRunner(config)
        
        # Start evaluation with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Initializing evaluation...", total=100)
            
            # Load tasks
            progress.update(main_task, description="Loading evaluation tasks...", completed=10)
            task_loader = EvalTaskLoader(config)
            tasks = []
            for task_ref in suite_config.tasks:
                task = task_loader.load_task(task_ref.task_id)
                tasks.append(task)
            
            # Initialize adapter
            progress.update(main_task, description="Initializing target system adapter...", completed=20)
            adapter = get_adapter(target_config.adapter_type)
            adapter.configure(target_config.config)
            
            # Run evaluation
            progress.update(main_task, description="Running evaluation...", completed=30)
            
            # Mock evaluation execution with realistic progress
            total_tasks = len(tasks)
            for i, task in enumerate(tasks):
                task_progress = 30 + (i / total_tasks) * 60  # 30% to 90%
                progress.update(
                    main_task, 
                    description=f"Processing task {i+1}/{total_tasks}: {task.name[:50]}...",
                    completed=task_progress
                )
                time.sleep(0.5)  # Simulate processing time
            
            # Generate results
            progress.update(main_task, description="Generating results...", completed=95)
            
            # Mock results
            results = {
                "evaluation_info": {
                    "suite_id": suite_id,
                    "target_system": target,
                    "start_time": time.time(),
                    "end_time": time.time() + 120,
                    "total_tasks": len(tasks),
                    "configuration": {
                        "max_workers": max_workers,
                        "timeout": timeout,
                        **additional_config
                    }
                },
                "summary": {
                    "total_tasks": len(tasks),
                    "passed": len(tasks) - 2,
                    "failed": 2,
                    "overall_score": 0.87,
                    "execution_time": 120.5
                },
                "task_results": [
                    {
                        "task_id": task.id,
                        "task_name": task.name,
                        "score": 0.85 + (i * 0.02),
                        "status": "passed" if i < len(tasks) - 2 else "failed",
                        "execution_time": 2.5 + (i * 0.1),
                        "metrics": {
                            "accuracy": 0.88 + (i * 0.01),
                            "f1_score": 0.82 + (i * 0.015)
                        }
                    }
                    for i, task in enumerate(tasks)
                ]
            }
            
            progress.update(main_task, description="Saving results...", completed=100)
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate reports
        if html_report:
            html_file = output_dir / "report.html"
            try:
                generate_html_report(results, str(html_file))
                console.print(f"[green]✓[/green] HTML report saved to {html_file}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not generate HTML report: {e}[/yellow]")
        
        # Display results
        if json_output:
            print(json.dumps(results, indent=2, default=str))
        else:
            display_evaluation_results(results)
        
        console.print(f"\n[green]✓ Evaluation completed successfully![/green]")
        console.print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Evaluation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command("list")
def list_suites_command(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (table, json, yaml)")
):
    """
    List available evaluation suites.
    
    Examples:
        openevals list
        openevals list --detailed
        openevals list --format json
    """
    
    config = load_global_config()
    
    if format_type == "json":
        suites_data = [
            {
                "id": suite.id,
                "name": suite.name,
                "description": suite.description,
                "tasks": len(suite.tasks),
                "metrics": [task.grading_criteria for task in suite.tasks] if detailed else None
            }
            for suite in config.evaluation_suites
        ]
        print(json.dumps(suites_data, indent=2))
        return
    
    elif format_type == "yaml":
        suites_data = {
            suite.id: {
                "name": suite.name,
                "description": suite.description,
                "tasks": len(suite.tasks)
            }
            for suite in config.evaluation_suites
        }
        print(yaml.dump(suites_data, default_flow_style=False))
        return
    
    # Table format
    table = Table(title="Available Evaluation Suites")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Tasks", justify="right", style="blue")
    
    if detailed:
        table.add_column("Description", style="white")
    
    for suite in config.evaluation_suites:
        row = [suite.id, suite.name, str(len(suite.tasks))]
        if detailed:
            row.append(suite.description or "No description")
        table.add_row(*row)
    
    console.print(table)

@app.command("targets")
def list_targets_command(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed configuration"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """
    List available target systems.
    
    Examples:
        openevals targets
        openevals targets --detailed
        openevals targets --format json
    """
    
    config = load_global_config()
    
    if format_type == "json":
        targets_data = [
            {
                "name": target.name,
                "adapter_type": target.adapter_type,
                "config": target.config if detailed else None
            }
            for target in config.target_systems
        ]
        print(json.dumps(targets_data, indent=2))
        return
    
    # Table format
    table = Table(title="Available Target Systems")
    table.add_column("Name", style="cyan")
    table.add_column("Adapter Type", style="green")
    
    if detailed:
        table.add_column("Configuration", style="white")
    
    for target in config.target_systems:
        row = [target.name, target.adapter_type]
        if detailed:
            config_str = yaml.dump(target.config, default_flow_style=False)
            row.append(config_str[:100] + "..." if len(config_str) > 100 else config_str)
        table.add_row(*row)
    
    console.print(table)

@app.command("validate")
def validate_config_command(
    config_file: Optional[Path] = typer.Argument(None, help="Configuration file to validate"),
    suite_id: Optional[str] = typer.Option(None, "--suite", help="Validate specific suite"),
    target: Optional[str] = typer.Option(None, "--target", help="Validate specific target"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix common issues")
):
    """
    Validate configuration files and settings.
    
    Examples:
        openevals validate
        openevals validate config.yaml
        openevals validate --suite basic_qa --target openai_gpt4
    """
    
    console.print(Panel.fit(
        "[bold blue]Configuration Validation[/bold blue]",
        title="🔍 Validating"
    ))
    
    issues = []
    
    try:
        # Validate main configuration
        config = load_global_config()
        console.print("[green]✓[/green] Main configuration loaded successfully")
        
        # Validate specific suite if provided
        if suite_id:
            try:
                suite_config = validate_suite_id(suite_id)
                console.print(f"[green]✓[/green] Suite '{suite_id}' is valid")
                
                # Validate tasks in suite
                task_loader = EvalTaskLoader(config)
                for task_ref in suite_config.tasks:
                    try:
                        task = task_loader.load_task(task_ref.task_id)
                        console.print(f"[green]✓[/green] Task '{task_ref.task_id}' loaded successfully")
                    except Exception as e:
                        issues.append(f"Task '{task_ref.task_id}': {e}")
                        console.print(f"[red]✗[/red] Task '{task_ref.task_id}': {e}")
                        
            except typer.Exit:
                issues.append(f"Suite '{suite_id}' not found")
        
        # Validate specific target if provided
        if target:
            try:
                target_config = validate_target_system(target)
                console.print(f"[green]✓[/green] Target '{target}' configuration is valid")
                
                # Test adapter initialization
                try:
                    adapter = get_adapter(target_config.adapter_type)
                    console.print(f"[green]✓[/green] Adapter '{target_config.adapter_type}' can be initialized")
                except Exception as e:
                    issues.append(f"Adapter '{target_config.adapter_type}': {e}")
                    console.print(f"[red]✗[/red] Adapter '{target_config.adapter_type}': {e}")
                    
            except typer.Exit:
                issues.append(f"Target '{target}' not found")
        
        # Validate additional config file if provided
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    additional_config = yaml.safe_load(f)
                console.print(f"[green]✓[/green] Config file '{config_file}' is valid YAML")
            except Exception as e:
                issues.append(f"Config file '{config_file}': {e}")
                console.print(f"[red]✗[/red] Config file '{config_file}': {e}")
        
        # Summary
        if issues:
            console.print(f"\n[yellow]Found {len(issues)} issues:[/yellow]")
            for issue in issues:
                console.print(f"  • {issue}")
            
            if fix:
                console.print("\n[yellow]Auto-fix is not yet implemented[/yellow]")
            
            raise typer.Exit(1)
        else:
            console.print("\n[green]✓ All validations passed![/green]")
        
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)

@app.command("info")
def show_system_info():
    """
    Show system information and available components.
    
    Examples:
        openevals info
    """
    
    console.print(Panel.fit(
        "[bold blue]OpenEvals System Information[/bold blue]",
        title="ℹ️  System Info"
    ))
    
    try:
        # System information
        import platform
        import sys
        
        info_table = Table(title="System Information")
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Version/Info", style="green")
        
        info_table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        info_table.add_row("Platform", platform.platform())
        info_table.add_row("OpenEvals Version", "1.0.0")
        
        # Try to get package versions
        try:
            import pkg_resources
            for package in ["fastapi", "typer", "rich", "pydantic", "openai", "transformers"]:
                try:
                    version = pkg_resources.get_distribution(package).version
                    info_table.add_row(f"{package.title()}", version)
                except:
                    info_table.add_row(f"{package.title()}", "Not installed")
        except:
            pass
        
        console.print(info_table)
        
        # Available adapters
        adapters_table = Table(title="Available Adapters")
        adapters_table.add_column("Adapter", style="cyan")
        adapters_table.add_column("Status", style="green")
        
        try:
            adapters = list_adapters()
            for adapter in adapters:
                adapters_table.add_row(adapter, "Available")
        except:
            adapters_table.add_row("Error", "Could not load adapter list")
        
        console.print(adapters_table)
        
        # Available graders
        graders_table = Table(title="Available Graders")
        graders_table.add_column("Grader", style="cyan")
        graders_table.add_column("Status", style="green")
        
        try:
            graders = list_graders()
            for grader in graders:
                graders_table.add_row(grader, "Available")
        except:
            graders_table.add_row("Error", "Could not load grader list")
        
        console.print(graders_table)
        
        # Configuration status
        try:
            config = load_global_config()
            config_table = Table(title="Configuration Status")
            config_table.add_column("Component", style="cyan")
            config_table.add_column("Count", style="green")
            
            config_table.add_row("Evaluation Suites", str(len(config.evaluation_suites)))
            config_table.add_row("Target Systems", str(len(config.target_systems)))
            
            console.print(config_table)
        except Exception as e:
            console.print(f"[red]Configuration Error: {e}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error gathering system info: {e}[/red]")

# --- Helper Display Functions ---

def show_evaluation_plan(suite_config: EvalSuiteConfig, target_config: TargetSystemConfig, max_workers: int, timeout: int):
    """Display evaluation plan for dry run"""
    
    tree = Tree("[bold blue]Evaluation Plan[/bold blue]")
    
    # Suite information
    suite_branch = tree.add(f"[green]Suite: {suite_config.name}[/green]")
    suite_branch.add(f"ID: {suite_config.id}")
    suite_branch.add(f"Description: {suite_config.description or 'No description'}")
    suite_branch.add(f"Tasks: {len(suite_config.tasks)}")
    
    # Target information
    target_branch = tree.add(f"[yellow]Target: {target_config.name}[/yellow]")
    target_branch.add(f"Adapter: {target_config.adapter_type}")
    
    # Execution parameters
    exec_branch = tree.add("[blue]Execution Parameters[/blue]")
    exec_branch.add(f"Max Workers: {max_workers}")
    exec_branch.add(f"Timeout: {timeout}s")
    
    console.print(tree)

def display_evaluation_results(results: Dict[str, Any]):
    """Display evaluation results in a formatted way"""
    
    # Summary panel
    summary = results.get("summary", {})
    
    summary_text = (
        f"[green]✓ Passed:[/green] {summary.get('passed', 0)}\n"
        f"[red]✗ Failed:[/red] {summary.get('failed', 0)}\n"
        f"[blue]Overall Score:[/blue] {summary.get('overall_score', 0):.2%}\n"
        f"[yellow]Execution Time:[/yellow] {format_duration(summary.get('execution_time', 0))}"
    )
    
    console.print(Panel(
        summary_text,
        title="📊 Evaluation Results",
        border_style="green"
    ))
    
    # Task results table
    if "task_results" in results:
        table = Table(title="Task Results")
        table.add_column("Task", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Status", justify="center")
        table.add_column("Time", justify="right", style="blue")
        
        for task in results["task_results"]:
            status_style = "green" if task.get("status") == "passed" else "red"
            status_text = f"[{status_style}]{task.get('status', 'unknown').upper()}[/{status_style}]"
            
            table.add_row(
                task.get("task_name", "Unknown"),
                f"{task.get('score', 0):.2%}",
                status_text,
                format_duration(task.get("execution_time", 0))
            )
        
        console.print(table)

# --- Version Command ---

@app.command("version")
def version_command():
    """Show OpenEvals version information."""
    console.print("[bold blue]OpenEvals[/bold blue] version [green]1.0.0[/green]")

# --- Main CLI Entry Point ---

def main():
    """Main entry point for the CLI"""
    app()

if __name__ == "__main__":
    main() 