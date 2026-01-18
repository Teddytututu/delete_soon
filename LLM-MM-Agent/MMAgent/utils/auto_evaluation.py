"""
Automatic Quality Evaluation for MM-Agent

Automatically evaluates the quality of MM-Agent output after each run.
Generates detailed reports identifying problems and areas for improvement.

Author: MM-Agent Team
Date: 2026-01-09
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Iterable


class AutoEvaluator:
    """
    Automatic evaluator for MM-Agent outputs.

    Assesses quality across multiple dimensions and identifies specific problems.
    """

    def __init__(self, llm, output_dir: str, problem_id: str):
        """Initialize evaluator.

        Args:
            llm: Language model instance
            output_dir: Output directory path
            problem_id: Problem identifier
        """
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.problem_id = problem_id
        self.evaluation_results = {
            'problem_id': problem_id,
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'overall_quality_score': 0.0,
            'problems_identified': [],
            'recommendations': []
        }

    def _iter_tasks(self, solution: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """
        CRITICAL FIX: Iterate over tasks regardless of whether they're stored as list or dict.

        computational_solving.py treats tasks as a LIST (solution['tasks'].append()),
        but some evaluation code tried to use .values() or .items() which only works on dicts.

        This helper handles both cases for backward compatibility.

        Args:
            solution: Solution dictionary with 'tasks' key

        Yields:
            Tuples of (task_id, task_data) where task_id is 1-indexed
        """
        tasks = solution.get("tasks", [])

        if isinstance(tasks, dict):
            # Legacy format: dict with numeric or string keys
            for key in sorted(tasks.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
                task_id = int(str(key)) if str(key).isdigit() else key
                yield task_id, tasks[key]
        elif isinstance(tasks, list):
            # Current format: list of task dicts
            for i, task in enumerate(tasks, start=1):
                if isinstance(task, dict):
                    yield i, task
        else:
            # Unknown format - return empty iterator
            return

    def evaluate_output(self) -> str:
        """
        Evaluate the complete MM-Agent output.

        Returns:
            Path to generated evaluation report
        """
        print("\n" + "="*80)
        print("AUTOMATIC QUALITY EVALUATION")
        print("="*80)
        print(f"[INFO] Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[INFO] Evaluator output_dir = {self.output_dir.resolve()}")

        # Step 1: Check if solution JSON exists
        print("\n[1/6] Checking for solution file...")
        # CRITICAL FIX: Use problem_id instead of hardcoded "solution.json"
        # The actual file is named {problem_id}.json (e.g., "2025_C.json")
        solution_path = self.output_dir / "json" / f"{self.problem_id}.json"
        print(f"      Looking for: {solution_path}")

        if not solution_path.exists():
            print(f"[WARNING] Solution file not found: {solution_path}")
            print(f"[INFO] This may indicate that the pipeline failed before completion")
            print(f"[INFO] Checking what files ARE available in {self.output_dir}/json/...")

            json_dir = self.output_dir / "json"
            if json_dir.exists():
                files = list(json_dir.glob("*"))
                print(f"[INFO] Found {len(files)} files: {[f.name for f in files]}")
            else:
                print(f"[WARNING] json/ directory doesn't exist at all")

            return self._generate_minimal_report("Solution file not found")

        print(f"[OK] Solution file exists")

        # Step 2: Load solution
        print("\n[2/6] Loading solution file...")
        try:
            with open(solution_path, 'r', encoding='utf-8') as f:
                solution = json.load(f)
            file_size = solution_path.stat().st_size
            print(f"[OK] Loaded solution from {solution_path}")
            print(f"[INFO] File size: {file_size:,} bytes")
            print(f"[INFO] Solution keys: {list(solution.keys())}")
        except Exception as e:
            print(f"[ERROR] Failed to load solution: {e}")
            return self._generate_minimal_report(f"Failed to load solution: {e}")

        # Step 3: Evaluate Problem Analysis
        print("\n[3/6] Evaluating Problem Analysis stage...")
        self._evaluate_problem_analysis(solution)

        # Step 4: Evaluate Mathematical Modeling
        print("\n[4/6] Evaluating Mathematical Modeling stage...")
        self._evaluate_mathematical_modeling(solution)

        # Step 5: Evaluate Computational Solving
        print("\n[5/6] Evaluating Computational Solving stage...")
        self._evaluate_computational_solving(solution)

        # Step 6: Evaluate Solution Reporting
        print("\n[6/6] Evaluating Solution Reporting stage...")
        self._evaluate_solution_reporting(solution)

        # Calculate overall score
        print("\nCalculating overall quality score...")
        if self.evaluation_results['stages']:
            scores = [s.get('quality_score', 0) for s in self.evaluation_results['stages'].values()]
            self.evaluation_results['overall_quality_score'] = sum(scores) / len(scores)
            print(f"[OK] Overall score calculated: {self.evaluation_results['overall_quality_score']:.2f}")
        else:
            print("[WARNING] No stage scores available, using default 0.0")
            self.evaluation_results['overall_quality_score'] = 0.0

        # Generate report
        print("\nGenerating evaluation report...")
        report_path = self._generate_report()
        print(f"[OK] Report generated: {report_path}")

        # Print summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Overall Quality Score: {self.evaluation_results['overall_quality_score']:.2f}/1.00")
        print(f"Problems Identified: {len(self.evaluation_results['problems_identified'])}")

        if self.evaluation_results['problems_identified']:
            print("\nProblems by severity:")
            high_severity = [p for p in self.evaluation_results['problems_identified'] if p['severity'] == 'HIGH']
            medium_severity = [p for p in self.evaluation_results['problems_identified'] if p['severity'] == 'MEDIUM']
            print(f"  HIGH: {len(high_severity)}")
            print(f"  MEDIUM: {len(medium_severity)}")

        print(f"\nReport saved to: {report_path}")
        print(f"JSON data saved to: {self.output_dir}/evaluation/quality_data.json")
        print("="*80 + "\n")

        return str(report_path)

    def _evaluate_problem_analysis(self, solution: Dict[str, Any]):
        """Evaluate Problem Analysis stage."""
        print("\n[1/4] Evaluating Problem Analysis...")

        stage_result = {
            'stage_name': 'Problem Analysis',
            'quality_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'problems': []
        }

        # Check for required components
        if 'task_analyses' in solution or 'tasks' in solution:
            stage_result['quality_score'] += 0.3
            stage_result['strengths'].append('Task decomposition present')

        if 'background' in solution or 'problem_background' in solution:
            stage_result['quality_score'] += 0.2
            stage_result['strengths'].append('Problem background understood')

        if 'variables' in solution or 'variable_description' in solution:
            stage_result['quality_score'] += 0.2
            stage_result['strengths'].append('Variables identified')

        # Check for common problems with absolute path evidence
        if not solution.get('task_analyses') and not solution.get('tasks'):
            stage_result['problems'].append({
                'category': 'Completeness',
                'severity': 'HIGH',
                'description': 'No task decomposition found',
                'evidence': f'Solution file: {self.output_dir / "json" / "solution.json"} | Score: {stage_result["quality_score"]:.2f}'
            })
            stage_result['weaknesses'].append('Missing task breakdown')

        # CRITICAL FIX: Aggregate stage problems to global problems_identified
        for problem in stage_result['problems']:
            self.evaluation_results['problems_identified'].append({
                'stage': 'problem_analysis',
                **problem
            })

        self.evaluation_results['stages']['problem_analysis'] = stage_result

        score = stage_result['quality_score']
        status = "[OK]" if score >= 0.7 else "[WARNING]"
        print(f"      {status} Quality Score: {score:.2f}")

    def _evaluate_mathematical_modeling(self, solution: Dict[str, Any]):
        """Evaluate Mathematical Modeling stage."""
        print("\n[2/4] Evaluating Mathematical Modeling...")

        stage_result = {
            'stage_name': 'Mathematical Modeling',
            'quality_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'problems': []
        }

        # Check for modeling components
        if 'tasks' in solution:
            # CRITICAL FIX: Use _iter_tasks() to handle both list and dict formats
            for task_id, task_data in self._iter_tasks(solution):
                if 'modeling_formulas' in task_data or 'formulas' in task_data:
                    stage_result['quality_score'] += 0.3
                    stage_result['strengths'].append(f'Task {task_id}: Mathematical formulas present')
                    break

                if 'modeling_method' in task_data or 'method' in task_data:
                    stage_result['quality_score'] += 0.2
                    stage_result['strengths'].append(f'Task {task_id}: Modeling method specified')
                    break

        # Check for HMML usage with deep inspection and evidence
        hmml_used = False
        hmml_evidence = []

        # Check tasks for HMML method retrieval
        if 'tasks' in solution:
            # CRITICAL FIX: Use _iter_tasks() to handle both list and dict formats
            for task_id, task_data in self._iter_tasks(solution):
                # Deep inspection: Check for method_retrieval or retrieved_methods
                if 'method_retrieval' in task_data:
                    hmml_used = True
                    methods = task_data.get('method_retrieval', [])
                    if isinstance(methods, list) and methods:
                        method_names = [m.get('method_name', m.get('name', 'unknown')) for m in methods[:3]]
                        hmml_evidence.append(f"Task {task_id}: Retrieved {len(methods)} methods: {', '.join(method_names)}")
                    elif isinstance(methods, dict):
                        hmml_evidence.append(f"Task {task_id}: Retrieved methods with keys {list(methods.keys())}")

                # Check for modeling_method with HMML indicators
                if 'modeling_method' in task_data:
                    method_info = task_data['modeling_method']
                    if isinstance(method_info, dict):
                        if any(k in method_info for k in ['hmml_id', 'hmml_path', 'retrieved_from']):
                            hmml_used = True
                            hmml_evidence.append(f"Task {task_id}: HMML method with metadata {list(method_info.keys())}")

        # Also check solution-level HMML references
        solution_str = json.dumps(solution, ensure_ascii=False)
        if 'HMML' in solution_str:
            # Extract evidence from solution
            if 'hmml' in solution_str.lower():
                hmml_used = True
                if not hmml_evidence:
                    hmml_evidence.append("HMML references found in solution structure")

        if hmml_used:
            stage_result['quality_score'] += 0.2
            stage_result['strengths'].append('Used HMML knowledge base')
            # Add evidence to strengths for debugging
            for evidence in hmml_evidence[:3]:  # Limit to first 3 evidences
                stage_result['strengths'].append(f'Evidence: {evidence[:100]}...' if len(evidence) > 100 else f'Evidence: {evidence}')

        # Check for problems with absolute path evidence
        if stage_result['quality_score'] < 0.5:
            stage_result['problems'].append({
                'category': 'Method Selection',
                'severity': 'MEDIUM',
                'description': 'Mathematical modeling quality appears low',
                'evidence': f'Solution file: {self.output_dir / "json" / "solution.json"} | Score: {stage_result["quality_score"]:.2f}'
            })
            stage_result['weaknesses'].append('Insufficient mathematical rigor')

        # CRITICAL FIX: Aggregate stage problems to global problems_identified
        for problem in stage_result['problems']:
            self.evaluation_results['problems_identified'].append({
                'stage': 'mathematical_modeling',
                **problem
            })

        self.evaluation_results['stages']['mathematical_modeling'] = stage_result

        score = stage_result['quality_score']
        status = "[OK]" if score >= 0.7 else "[WARNING]"
        print(f"      {status} Quality Score: {score:.2f}")

    def _evaluate_computational_solving(self, solution: Dict[str, Any]):
        """Evaluate Computational Solving stage."""
        print("\n[3/4] Evaluating Computational Solving...")

        stage_result = {
            'stage_name': 'Computational Solving',
            'quality_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'problems': []
        }

        # Check for code execution results
        code_dir = self.output_dir / "code"
        if code_dir.exists():
            code_files = list(code_dir.glob("*.py"))
            if code_files:
                stage_result['quality_score'] += 0.3
                stage_result['strengths'].append(f'Generated {len(code_files)} Python files')

        # Check for results
        if 'tasks' in solution:
            results_count = 0
            # CRITICAL FIX: Use _iter_tasks() to handle both list and dict formats
            for task_id, task_data in self._iter_tasks(solution):
                # CRITICAL FIX: Check for actual field names generated by computational_solving.py
                # Old buggy code checked for 'results'/'output' which don't exist
                # New code checks for: 'execution_result', 'solution_interpretation', 'subtask_outcome_analysis'
                if 'execution_result' in task_data or 'solution_interpretation' in task_data or 'subtask_outcome_analysis' in task_data:
                    results_count += 1

            if results_count > 0:
                stage_result['quality_score'] += 0.4
                stage_result['strengths'].append(f'Found results for {results_count} tasks')

        # CRITICAL FIX: Check for charts inside tasks, not at top level
        # Charts are stored within individual task data, not at solution root
        has_charts = False
        if 'tasks' in solution:
            for task_id, task_data in self._iter_tasks(solution):
                if 'charts' in task_data or any('chart' in str(k).lower() for k in task_data.keys()):
                    has_charts = True
                    break

        if has_charts:
            stage_result['quality_score'] += 0.3
            stage_result['strengths'].append('Generated visualizations/charts')

        # Check for problems with absolute path evidence
        if stage_result['quality_score'] < 0.5:
            stage_result['problems'].append({
                'category': 'Code Quality',
                'severity': 'HIGH',
                'description': 'Computational results appear incomplete',
                'evidence': f'Code dir: {self.output_dir / "code"} | Solution file: {self.output_dir / "json" / "solution.json"} | Score: {stage_result["quality_score"]:.2f}'
            })
            stage_result['weaknesses'].append('Missing or incomplete results')

        # CRITICAL FIX: Aggregate stage problems to global problems_identified
        for problem in stage_result['problems']:
            self.evaluation_results['problems_identified'].append({
                'stage': 'computational_solving',
                **problem
            })

        self.evaluation_results['stages']['computational_solving'] = stage_result

        score = stage_result['quality_score']
        status = "[OK]" if score >= 0.7 else "[WARNING]"
        print(f"      {status} Quality Score: {score:.2f}")

    def _evaluate_solution_reporting(self, solution: Dict[str, Any]):
        """Evaluate Solution Reporting stage."""
        print("\n[4/4] Evaluating Solution Reporting...")

        stage_result = {
            'stage_name': 'Solution Reporting',
            'quality_score': 0.0,
            'strengths': [],
            'weaknesses': [],
            'problems': []
        }

        # Check for report files
        # CRITICAL FIX: Support both {problem_id}.md (current) and solution.md (legacy)
        # [FIXED] Now looking in Workspace/ subdirectory (three-tier architecture)
        markdown_path = self.output_dir / "Workspace" / "markdown" / f"{self.problem_id}.md"
        markdown_path_alt = self.output_dir / "Workspace" / "markdown" / "solution.md"
        latex_path = self.output_dir / "Workspace" / "latex" / f"{self.problem_id}.tex"
        latex_path_alt = self.output_dir / "Workspace" / "latex" / "solution.tex"

        # Legacy paths for backward compatibility (check root level too)
        markdown_path_legacy = self.output_dir / "markdown" / f"{self.problem_id}.md"
        latex_path_legacy = self.output_dir / "latex" / f"{self.problem_id}.tex"

        if markdown_path.exists() or markdown_path_alt.exists() or markdown_path_legacy.exists():
            stage_result['quality_score'] += 0.4
            stage_result['strengths'].append('Markdown report generated')

        if latex_path.exists() or latex_path_alt.exists() or latex_path_legacy.exists():
            stage_result['quality_score'] += 0.4
            stage_result['strengths'].append('LaTeX report generated')

        # Check for report structure
        if 'conclusion' in str(solution).lower() or 'discussion' in str(solution).lower():
            stage_result['quality_score'] += 0.2
            stage_result['strengths'].append('Includes conclusion/discussion')

        # Check for problems with absolute path evidence
        has_markdown = markdown_path.exists() or markdown_path_alt.exists() or markdown_path_legacy.exists()
        has_latex = latex_path.exists() or latex_path_alt.exists() or latex_path_legacy.exists()
        if not has_markdown and not has_latex:
            stage_result['problems'].append({
                'category': 'Completeness',
                'severity': 'HIGH',
                'description': 'No report files generated',
                'evidence': f'Expected: {markdown_path} OR {latex_path} | Output dir: {self.output_dir}'
            })
            stage_result['weaknesses'].append('Missing solution report')

        # CRITICAL FIX: Aggregate stage problems to global problems_identified
        for problem in stage_result['problems']:
            self.evaluation_results['problems_identified'].append({
                'stage': 'solution_reporting',
                **problem
            })

        self.evaluation_results['stages']['solution_reporting'] = stage_result

        score = stage_result['quality_score']
        status = "[OK]" if score >= 0.7 else "[WARNING]"
        print(f"      {status} Quality Score: {score:.2f}")

    def _generate_report(self) -> Path:
        """Generate evaluation report file."""
        report_dir = self.output_dir / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "quality_report.txt"
        json_path = report_dir / "quality_data.json"

        # Generate human-readable report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MM-Agent AUTOMATIC QUALITY EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Problem ID: {self.problem_id}\n")
            f.write(f"Evaluation Time: {self.evaluation_results['timestamp']}\n")
            f.write(f"Overall Quality Score: {self.evaluation_results['overall_quality_score']:.2f}/1.00\n\n")

            # Stage results
            f.write("-"*80 + "\n")
            f.write("STAGE-BY-STAGE EVALUATION\n")
            f.write("-"*80 + "\n\n")

            for stage_name, stage_data in self.evaluation_results['stages'].items():
                f.write(f"\n{stage_name.replace('_', ' ').title()}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Quality Score: {stage_data['quality_score']:.2f}/1.00\n")

                if stage_data.get('strengths'):
                    f.write("\nStrengths:\n")
                    for strength in stage_data['strengths']:
                        f.write(f"  [+] {strength}\n")

                if stage_data.get('weaknesses'):
                    f.write("\nWeaknesses:\n")
                    for weakness in stage_data['weaknesses']:
                        f.write(f"  [-] {weakness}\n")

                if stage_data.get('problems'):
                    f.write("\nProblems Identified:\n")
                    for problem in stage_data['problems']:
                        f.write(f"  [{problem['severity']}] {problem['category']}: {problem['description']}\n")
                        # CRITICAL FIX: Don't append here - problems already added during evaluation phase
                        # Adding them here causes duplication and is a side effect

                f.write("\n")

            # Summary
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n\n")

            total_problems = len(self.evaluation_results['problems_identified'])
            f.write(f"Total Problems Identified: {total_problems}\n")

            if total_problems > 0:
                f.write("\nProblem Breakdown by Severity:\n")
                high_count = sum(1 for p in self.evaluation_results['problems_identified'] if p['severity'] == 'HIGH')
                medium_count = sum(1 for p in self.evaluation_results['problems_identified'] if p['severity'] == 'MEDIUM')
                f.write(f"  HIGH: {high_count}\n")
                f.write(f"  MEDIUM: {medium_count}\n")

                f.write("\nRecommendations:\n")
                for problem in self.evaluation_results['problems_identified'][:5]:
                    f.write(f"  1. Address {problem['stage']}: {problem['description']}\n")

            # Overall assessment
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL ASSESSMENT\n")
            f.write("="*80 + "\n\n")

            overall_score = self.evaluation_results['overall_quality_score']
            if overall_score >= 0.8:
                f.write("Status: EXCELLENT\n")
                f.write("The solution demonstrates high quality across all stages.\n")
            elif overall_score >= 0.6:
                f.write("Status: GOOD\n")
                f.write("The solution is acceptable but has room for improvement.\n")
            elif overall_score >= 0.4:
                f.write("Status: FAIR\n")
                f.write("The solution meets minimum requirements but needs significant improvement.\n")
            else:
                f.write("Status: POOR\n")
                f.write("The solution has major issues that need to be addressed.\n")

        # Save JSON data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        return report_path

    def _generate_minimal_report(self, error_message: str) -> str:
        """Generate minimal report when evaluation fails."""
        report_dir = self.output_dir / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "quality_report.txt"
        json_path = report_dir / "quality_data.json"

        # Update evaluation results with error
        self.evaluation_results['error'] = error_message
        self.evaluation_results['overall_quality_score'] = 0.0

        # Generate text report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MM-Agent AUTOMATIC QUALITY EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Problem ID: {self.problem_id}\n")
            f.write(f"Evaluation Time: {self.evaluation_results['timestamp']}\n")
            f.write(f"Status: EVALUATION FAILED\n\n")

            f.write(f"Error: {error_message}\n")
            f.write("\nUnable to complete automatic evaluation.\n")
            f.write("Please check if the solution was generated successfully.\n")

        # Generate JSON data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        return str(report_path)


def run_auto_evaluation(llm, output_dir: str, problem_id: str) -> str:
    """
    Run automatic evaluation on MM-Agent output.

    Args:
        llm: Language model instance
        output_dir: Output directory path
        problem_id: Problem identifier

    Returns:
        Path to generated evaluation report
    """
    evaluator = AutoEvaluator(llm, output_dir, problem_id)
    return evaluator.evaluate_output()
