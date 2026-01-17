from collections import deque
from prompt.template import TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT, TASK_DEPENDENCY_ANALYSIS_PROMPT, DAG_CONSTRUCTION_PROMPT, CODE_STRUCTURE_PROMPT
import json
import sys
import re
import logging  # [NEW] For structured logging
# CRITICAL FIX: Import robust_json_load for unified JSON parsing
from utils.utils import robust_json_load
# [NEW] Add explicit type hints
from typing import Dict, List, Union, Any, Optional


class Coordinator:
    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize Coordinator with optional logger.

        Args:
            llm: LLM instance for generating responses
            logger: Optional logger instance. If not provided, creates a default logger.
        """
        self.llm = llm
        self.memory: Dict[str, Any] = {}
        self.code_memory: Dict[str, Dict] = {}
        self.DAG: Dict[str, List[str]] = {}
        # [NEW] Use provided logger or create default for backward compatibility
        self.logger = logger or logging.getLogger("Coordinator")

    # ========================================================================
    # Major Fix #10: Type Normalization Helper
    # ========================================================================

    def _normalize_graph(self, graph: Dict[Any, Any]) -> Dict[str, List[str]]:
        """
        Major Fix #10: Enforce strict string typing for the DAG.

        Converts various LLM output formats to Dict[str, List[str]]:
        - {1: [2, "3"]} -> {"1": ["2", "3"]}
        - {"1": 2} -> {"1": ["2"]}
        - {"1": None} -> {"1": []}

        This eliminates KeyError and ensures type consistency throughout the system.

        Args:
            graph: Raw DAG from LLM (may have mixed types: int, str, single items)

        Returns:
            Normalized DAG with strict typing: Dict[str, List[str]]

        Examples:
            >>> _normalize_graph({1: [2, "3"]})
            {"1": ["2", "3"]}

            >>> _normalize_graph({"1": 2})
            {"1": ["2"]}
        """
        normalized: Dict[str, List[str]] = {}

        for node, dependencies in graph.items():
            # Step 1: Force Key to String
            node_str = str(node).strip()

            # Step 2: Force Value to List of Strings
            deps_str_list: List[str] = []

            if isinstance(dependencies, list):
                # Standard case: list of dependencies
                deps_str_list = [str(dep).strip() for dep in dependencies]
            elif isinstance(dependencies, (str, int, float)):
                # Edge case: LLM returned single item instead of list
                deps_str_list = [str(dependencies).strip()]
            elif dependencies is None:
                # Edge case: No dependencies
                deps_str_list = []
            else:
                # Fallback: try to convert to list
                try:
                    deps_str_list = [str(dep).strip() for dep in list(dependencies)]
                except Exception:
                    # Last resort: empty list
                    deps_str_list = []

            normalized[node_str] = deps_str_list

        return normalized

    # ========================================================================

    def compute_dag_order(self, graph: Dict[Any, Any]) -> List[str]:
        """
        Compute the topological sorting (computation order) of a DAG.

        Major Fix #10: Now uses _normalize_graph() for type safety.

        Args:
            graph: DAG represented as an adjacency list (may have mixed types)

        Returns:
            A list of task IDs (as strings) in computation order

        Raises:
            ValueError: If graph has cycles
        """
        # Major Fix #10: Use centralized normalization helper
        # This replaces the old manual loop and ensures consistency
        graph = self._normalize_graph(graph)

        # BUG FIX 2.4: Auto-complete missing nodes instead of failing
        # LLM may generate dependencies on nodes not yet defined as keys
        all_nodes = set(graph.keys())
        missing_nodes = set()

        for node, dependencies in graph.items():
            for dep in dependencies:
                if dep not in all_nodes:
                    missing_nodes.add(dep)

        # Auto-add missing nodes with empty dependency lists
        if missing_nodes:
            # CRITICAL FIX: Handle both numeric and non-numeric keys
            try:
                sorted_missing = sorted(missing_nodes, key=lambda x: int(x) if x.isdigit() else x)
                self.logger.info(f"Auto-adding {len(missing_nodes)} missing node(s) to graph: {sorted_missing}")
            except:
                self.logger.info(f"Auto-adding {len(missing_nodes)} missing node(s) to graph: {list(missing_nodes)}")
            for missing_node in missing_nodes:
                graph[missing_node] = []

        # PERFORMANCE FIX: Optimize from O(N^2) to O(V+E) by building reverse adjacency list
        # Build reverse adjacency list (which nodes depend on each node)
        dependents = {node: [] for node in graph}
        for node, dependencies in graph.items():
            for dep in dependencies:
                if dep in dependents:
                    dependents[dep].append(node)

        # Calculate indegree
        in_degree = {node: 0 for node in graph}
        for node in graph:
            in_degree[node] = len(graph[node])

        # Find all nodes with in-degree 0 (which can be used as the starting point for calculation)
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            # O(E) operation: Iterate through direct dependents only
            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check if there is a loop (if the number of sorted nodes is less than the total number of nodes, then there is a loop)
        if len(order) != len(graph):
            raise ValueError("Graph contains a cycle!")

        return order

    def analyze(self, num_tasks: int, modeling_problem: str, problem_analysis: str, modeling_solution: str, task_descriptions: str, with_code: bool):
        if with_code:
            prompt = TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT.format(num_tasks=num_tasks, modeling_problem=modeling_problem, problem_analysis=problem_analysis, modeling_solution=modeling_solution, task_descriptions=task_descriptions).strip()
        else:
            prompt = TASK_DEPENDENCY_ANALYSIS_PROMPT.format(num_tasks=num_tasks, modeling_problem=modeling_problem, problem_analysis=problem_analysis, modeling_solution=modeling_solution, task_descriptions=task_descriptions).strip()
        return self.llm.generate(prompt)

    def dag_construction(self, num_tasks: int, modeling_problem: str, problem_analysis: str, modeling_solution: str, task_descriptions: str, task_dependency_analysis: str, cycle_warning: str = ''):
        """
        Generate task dependency DAG using LLM.

        Args:
            cycle_warning: Optional warning message to inject into prompt when cycles detected
        """
        prompt = DAG_CONSTRUCTION_PROMPT.format(num_tasks=num_tasks, modeling_problem=modeling_problem, problem_analysis=problem_analysis, modeling_solution=modeling_solution, task_descriptions=task_descriptions, task_dependency_analysis=task_dependency_analysis).strip()

        # CRITICAL FIX: Inject cycle warning into prompt if provided
        if cycle_warning:
            prompt = f"""{prompt}

CRITICAL REQUIREMENT:
{cycle_warning}

You MUST ensure the resulting dependency graph is acyclic (no circular dependencies).
"""

        return self.llm.generate(prompt)

    def analyze_dependencies(self, modeling_problem: str, problem_analysis: str, modeling_solution: str, task_descriptions: str, num_tasks: int, with_code: bool):
        """
        CRITICAL FIX: Simplified dependency analysis using robust_json_load().

        Previous implementation had 100+ lines of Tier 1/2/3 fix methods with nested
        try-except blocks. This version uses the unified robust_json_load() function
        which handles all LLM JSON formatting issues through json_repair library.

        Changes:
        - Removed _apply_tier1_fixes(), _apply_tier2_fixes(), _apply_tier3_fixes()
        - Reduced retry attempts from 5 to 3 (higher success rate with json_repair)
        - Simplified error handling with single try-except per attempt
        """
        task_dependency_analysis = self.analyze(num_tasks, modeling_problem, problem_analysis, modeling_solution, task_descriptions, with_code)
        self.task_dependency_analysis = task_dependency_analysis.split('\n\n')
        last_error = None
        cycle_warning = ''

        # CRITICAL FIX: Reduced from 5 to 3 attempts since robust_json_load has higher success rate
        for attempt in range(1, 4):
            try:
                dependency_DAG_str = self.dag_construction(
                    num_tasks, modeling_problem, problem_analysis, modeling_solution,
                    task_descriptions, task_dependency_analysis, cycle_warning
                )

                self.logger.debug(f"Attempt {attempt}: Parsing DAG JSON...")

                # CRITICAL FIX: Use robust_json_load instead of Tier 1/2/3 fixes
                # This single function handles:
                # - Markdown code block extraction
                # - Trailing commas
                # - Missing quotes
                # - Comments
                # - And uses json_repair library for advanced fixes
                self.DAG = robust_json_load(dependency_DAG_str)

                if not isinstance(self.DAG, dict):
                    raise ValueError(f"Parsed DAG is not a dictionary, got {type(self.DAG)}")

                # Major Fix #10: Use centralized type normalization
                # This replaces the old manual loop and ensures consistency
                self.DAG = self._normalize_graph(self.DAG)

                self.logger.info(f"DAG JSON parsed successfully on attempt {attempt}")

                # Check for cycles
                order = self.compute_dag_order(self.DAG)
                self.logger.info(f"DAG validated (acyclic) on attempt {attempt}")
                return order

            except ValueError as e:
                # CRITICAL FIX: Specifically detect cycle errors for retry
                if "Graph contains a cycle" in str(e):
                    self.logger.warning(f"Attempt {attempt}: DAG contains cycle - retrying...")
                    last_error = e
                    cycle_warning = f"""PREVIOUS ATTEMPT FAILED: The graph contained cycles.
                    Invalid Cyclic Graph: {self.DAG}
                    Error: {str(e)}
                    Please ensure the graph is a Directed Acyclic Graph (DAG)."""
                    continue
                else:
                    # Other parsing errors - log and retry
                    self.logger.warning(f"Attempt {attempt} failed: {e}")
                    last_error = e
                    continue
            except Exception as e:
                self.logger.error(f"Unexpected error in DAG parsing: {e}")
                last_error = e
                continue

        self.logger.error("All DAG parsing attempts failed")
        raise last_error or Exception("DAG parsing failed")
