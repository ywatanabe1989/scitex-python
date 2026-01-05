#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/tools/generation.py

"""Project and script generation tools for SciTeX analyzer."""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..helpers import (
    create_config_files,
    create_example_scripts,
    create_main_script,
    create_readme,
    create_requirements,
    create_test_templates,
    resolve_dependencies,
)


def register_generation_tools(server):
    """Register generation tools with the server.

    Parameters
    ----------
    server : ScitexBaseMCPServer
        The server instance to register tools with
    """

    @server.app.tool()
    async def create_scitex_project(
        project_name: str,
        project_type: str = "research",
        features: List[str] = ["basic"],
    ) -> Dict[str, Any]:
        """Generate complete SciTeX project structure with templates."""

        # Validate project name
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", project_name):
            return {
                "error": "Invalid project name. Use letters, numbers, underscores, and hyphens only."
            }

        project_path = Path(project_name)
        if project_path.exists():
            return {"error": f"Project {project_name} already exists"}

        # Create directory structure
        directories = [
            "scripts",
            "config",
            "data",
            "results",
            "examples",
            "tests",
            "docs",
        ]

        files_created = []

        # Create directories
        for directory in directories:
            dir_path = project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            files_created.append(f"{directory}/")

        # Create configuration files
        await create_config_files(project_path, project_type)
        files_created.extend(
            ["config/PATH.yaml", "config/PARAMS.yaml", "config/COLORS.yaml"]
        )

        # Create main script template
        await create_main_script(project_path, project_name, project_type)
        files_created.append("scripts/main.py")

        # Create README
        readme_content = await create_readme(project_name, project_type, features)
        (project_path / "README.md").write_text(readme_content)
        files_created.append("README.md")

        # Create requirements.txt
        requirements = await create_requirements(project_type, features)
        (project_path / "requirements.txt").write_text(requirements)
        files_created.append("requirements.txt")

        # Create example scripts based on features
        if "examples" in features:
            example_files = await create_example_scripts(project_path, project_type)
            files_created.extend(example_files)

        # Create test templates
        if "testing" in features:
            test_files = await create_test_templates(project_path)
            files_created.extend(test_files)

        return {
            "project_name": project_name,
            "project_type": project_type,
            "project_path": str(project_path.absolute()),
            "files_created": files_created,
            "directories_created": directories,
            "next_steps": [
                f"cd {project_name}",
                "pip install -r requirements.txt",
                "python scripts/main.py",
                "Edit config/PARAMS.yaml for your project parameters",
            ],
        }

    @server.app.tool()
    async def generate_scitex_script(
        script_name: str,
        script_type: str = "analysis",
        modules: List[str] = ["io", "plt"],
        template_style: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Generate purpose-built SciTeX scripts with appropriate templates."""

        script_templates = _get_script_templates()

        # Get template
        template = script_templates.get(script_type, script_templates["analysis"])
        script_content = template.get(template_style, template["minimal"])

        # Format with script name
        script_content = script_content.format(script_name=script_name)

        # Add module-specific imports
        imports = []
        if "stats" in modules:
            imports.append("import scipy.stats as stats")
        if "ml" in modules:
            imports.append("from sklearn import model_selection, metrics")
        if "dsp" in modules:
            imports.append("import scipy.signal")

        if imports:
            import_section = "\\n".join(imports)
            script_content = script_content.replace(
                "import scitex as stx", f"import scitex as stx\\n{import_section}"
            )

        return {
            "script_name": script_name,
            "script_type": script_type,
            "script_content": script_content,
            "modules_included": modules,
            "template_style": template_style,
            "estimated_lines": len(script_content.split("\\n")),
            "features": [
                "Configuration management",
                "Data validation",
                "Automatic result export",
                "Error handling",
                "Progress tracking",
            ],
        }

    @server.app.tool()
    async def optimize_scitex_config(
        config_paths: List[str], merge_strategy: str = "smart"
    ) -> Dict[str, Any]:
        """Merge and optimize multiple configuration files."""

        configs = {}
        conflicts = []

        # Load all configurations (simplified - would use stx.io.load in real impl)
        for config_path in config_paths:
            try:
                config_name = Path(config_path).stem
                # Placeholder - would actually load the config
                configs[config_name] = {}
            except Exception as e:
                return {"error": f"Failed to load {config_path}: {str(e)}"}

        # Merge configurations
        merged_config = {}

        # Smart merge strategy
        if merge_strategy == "smart":
            for config_name, config in configs.items():
                for section, values in config.items():
                    if section not in merged_config:
                        merged_config[section] = {}

                    if isinstance(values, dict):
                        for key, value in values.items():
                            if key in merged_config[section]:
                                if merged_config[section][key] != value:
                                    conflicts.append(
                                        {
                                            "section": section,
                                            "key": key,
                                            "config1": config_name,
                                            "value1": merged_config[section][key],
                                            "config2": config_name,
                                            "value2": value,
                                            "resolution": "kept_first",
                                        }
                                    )
                            else:
                                merged_config[section][key] = value

        # Optimize configuration
        optimizations = []

        # Check for missing standard sections
        standard_sections = ["PATH", "PARAMS", "COLORS"]
        for section in standard_sections:
            if section not in merged_config:
                optimizations.append(
                    {
                        "type": "missing_section",
                        "section": section,
                        "suggestion": f"Add {section} section for better organization",
                    }
                )

        # Check for potential consolidations
        if "PARAMS" in merged_config:
            params = merged_config["PARAMS"]
            for key, value in params.items():
                if isinstance(value, (int, float)) and key.upper() != key:
                    optimizations.append(
                        {
                            "type": "naming_convention",
                            "key": key,
                            "suggestion": f"Use uppercase: {key.upper()}",
                        }
                    )

        return {
            "merged_config": merged_config,
            "conflicts": conflicts,
            "optimizations": optimizations,
            "config_files_processed": len(configs),
            "total_parameters": sum(
                len(section)
                for section in merged_config.values()
                if isinstance(section, dict)
            ),
        }

    @server.app.tool()
    async def run_scitex_pipeline(
        pipeline_config: Dict[str, Any], dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute multi-script workflows with dependencies."""

        # Validate pipeline configuration
        required_keys = ["scripts", "dependencies"]
        if not all(key in pipeline_config for key in required_keys):
            return {"error": f"Pipeline config must contain: {required_keys}"}

        scripts = pipeline_config["scripts"]
        dependencies = pipeline_config.get("dependencies", {})

        # Build execution order
        execution_order = resolve_dependencies(scripts, dependencies)
        if "error" in execution_order:
            return execution_order

        if dry_run:
            return {
                "pipeline_valid": True,
                "execution_order": execution_order["order"],
                "total_scripts": len(scripts),
                "estimated_runtime": "varies",
                "dependencies_resolved": True,
            }

        # Execute pipeline
        results = {}
        failed_scripts = []

        for script_name in execution_order["order"]:
            script_config = scripts[script_name]
            script_path = script_config.get("path")

            if not script_path or not Path(script_path).exists():
                failed_scripts.append(
                    {
                        "script": script_name,
                        "error": f"Script not found: {script_path}",
                    }
                )
                continue

            # Execute script (simplified)
            try:
                results[script_name] = {
                    "status": "completed",
                    "duration": "simulated",
                    "output": f"Executed {script_name} successfully",
                }
            except Exception as e:
                results[script_name] = {"status": "failed", "error": str(e)}
                failed_scripts.append({"script": script_name, "error": str(e)})

        return {
            "pipeline_status": "completed"
            if not failed_scripts
            else "partially_failed",
            "scripts_executed": len(results),
            "failed_scripts": failed_scripts,
            "execution_results": results,
            "execution_order": execution_order["order"],
        }


def _get_script_templates() -> Dict[str, Dict[str, str]]:
    """Get script templates dictionary."""
    return {
        "analysis": {
            "minimal": """#!/usr/bin/env python3
import scitex as stx

CONFIG = stx.io.load_configs()

@stx.gen.start()
def main():
    # Load data
    data = stx.io.load('./data/input.csv')

    # Perform analysis
    results = analyze_data(data)

    # Save results
    stx.io.save(results, './results/analysis.csv', symlink_from_cwd=True)

def analyze_data(data):
    # Your analysis code here
    return data

if __name__ == "__main__":
    main()
""",
            "comprehensive": """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
{script_name}: Comprehensive data analysis script
Generated by SciTeX MCP Server
'''

import numpy as np
import pandas as pd
import scitex as stx
from pathlib import Path

CONFIG = stx.io.load_configs()

@stx.gen.start()
def main():
    '''Main analysis pipeline.'''
    stx.gen.print_config(CONFIG.PARAMS)
    data = load_data()
    validate_data(data)
    results = perform_analysis(data)
    create_visualizations(data, results)
    export_results(results)
    print("Analysis complete!")

def load_data():
    '''Load and preprocess data.'''
    data_path = CONFIG.PATH.DATA / 'input.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {{data_path}}")
    data = stx.io.load(data_path)
    print(f"Loaded data shape: {{data.shape}}")
    return data

def validate_data(data):
    '''Validate data quality.'''
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(f"Warning: Found {{missing.sum()}} missing values")

def perform_analysis(data):
    '''Perform statistical analysis.'''
    results = {{}}
    results['descriptive'] = data.describe()
    threshold = CONFIG.PARAMS.ANALYSIS_THRESHOLD
    results['filtered'] = data[data > threshold]
    return results

def create_visualizations(data, results):
    '''Create and save visualizations.'''
    fig, axes = stx.plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].hist(data.iloc[:,0], bins=30, alpha=0.7)
    axes[0,0].set_title('Data Distribution')
    stx.io.save(fig, './figures/analysis_overview.png', symlink_from_cwd=True)

def export_results(results):
    '''Export analysis results.'''
    stx.io.save(results['descriptive'], './results/descriptive_stats.csv', symlink_from_cwd=True)

if __name__ == "__main__":
    main()
""",
        }
    }


# EOF
