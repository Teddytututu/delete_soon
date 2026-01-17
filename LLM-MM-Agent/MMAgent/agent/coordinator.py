from collections import deque
from prompt.template import TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT, TASK_DEPENDENCY_ANALYSIS_PROMPT, DAG_CONSTRUCTION_PROMPT, CODE_STRUCTURE_PROMPT
import json
import sys
import re


class Coordinator:
    def __init__(self, llm):
        self.llm = llm
        self.memory = {}
        self.code_memory = {}

    def compute_dag_order(self, graph):
        """
        Compute the topological sorting (computation order) of a DAG.
        :param graph: DAG represented as an adjacency list, in the format of {node: [other nodes that this node depends on]}.
        :return: A list representing the computation order.
        :raises ValueError: If graph has cycles or missing dependencies.
        """
        # BUG FIX 2.3: Normalize all types to strings to handle LLM output inconsistencies
        # LLM may generate string keys with integer dependencies or vice versa
        normalized_graph = {}
        for node, dependencies in graph.items():
            # Convert node to string
            node_str = str(node)
            # Convert all dependencies to strings
            deps_str = [str(dep) for dep in dependencies]
            normalized_graph[node_str] = deps_str

        # Use normalized graph for all further processing
        graph = normalized_graph

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
                print(f"[INFO] Auto-adding {len(missing_nodes)} missing node(s) to graph: {sorted_missing}")
            except:
                print(f"[INFO] Auto-adding {len(missing_nodes)} missing node(s) to graph: {list(missing_nodes)}")
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

    def analyze(self, tasknum: int, modeling_problem: str, problem_analysis: str, modeling_solution: str, task_descriptions: str, with_code: bool):
        if with_code:
            prompt = TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT.format(tasknum=tasknum, modeling_problem=modeling_problem, problem_analysis=problem_analysis, modeling_solution=modeling_solution, task_descriptions=task_descriptions).strip()
        else:
            prompt = TASK_DEPENDENCY_ANALYSIS_PROMPT.format(tasknum=tasknum, modeling_problem=modeling_problem, problem_analysis=problem_analysis, modeling_solution=modeling_solution, task_descriptions=task_descriptions).strip()
        return self.llm.generate(prompt)

    def dag_construction(self, tasknum: int, modeling_problem: str, problem_analysis: str, modeling_solution: str, task_descriptions: str, task_dependency_analysis: str, cycle_warning: str = ''):
        """
        Generate task dependency DAG using LLM.

        Args:
            cycle_warning: Optional warning message to inject into prompt when cycles detected
        """
        prompt = DAG_CONSTRUCTION_PROMPT.format(tasknum=tasknum, modeling_problem=modeling_problem, problem_analysis=problem_analysis, modeling_solution=modeling_solution, task_descriptions=task_descriptions, task_dependency_analysis=task_dependency_analysis).strip()

        # CRITICAL FIX: Inject cycle warning into prompt if provided
        if cycle_warning:
            prompt = f"""{prompt}

CRITICAL REQUIREMENT:
{cycle_warning}

You MUST ensure the resulting dependency graph is acyclic (no circular dependencies).
"""

        return self.llm.generate(prompt)

    def _apply_tier1_fixes(self, json_str):
        """
        Tier 1 fixes: Always apply, low risk, high safety.
        - Extract code blocks
        - Find JSON boundaries
        - Remove trailing commas
        - Remove control characters
        """
        # Strategy 1: Remove ```json and ``` code blocks
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()

        # Strategy 2: Find JSON object boundaries
        json_match = re.search(r'\{[\s\S]*\}', json_str)
        if json_match:
            json_str = json_match.group()

        # Strategy 4: Remove trailing commas (common LLM error)
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove comma before }
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove comma before ]

        # Strategy 7: Remove control characters (low risk)
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')

        return json_str

    def _apply_tier2_fixes(self, json_str):
        """
        Tier 2 fixes: Medium risk, apply only when Tier 1 fails.
        - Remove comments
        - Fix unquoted keys
        """
        # Strategy 5: Remove JSON comments (// and /* */)
        json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove // comments
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)  # Remove /* */ comments

        # Strategy 6: Fix unquoted keys (common LLM error)
        # This regex finds patterns like {key: value} and converts to {"key": value}
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)

        return json_str

    def _apply_tier3_fixes(self, json_str):
        """
        Tier 3 fixes: High risk, apply only as last resort.
        - Replace single quotes with double quotes
        """
        # Strategy 3: Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')

        return json_str

    def analyze_dependencies(self, modeling_problem: str, problem_analysis: str, modeling_solution: str, task_descriptions: str, tasknum: int, with_code: bool):
        # CRITICAL FIX: task_descriptions is a string, len() would count characters not tasks
        # Pass tasknum explicitly instead
        task_dependency_analysis = self.analyze(tasknum, modeling_problem, problem_analysis, modeling_solution, task_descriptions, with_code)
        self.task_dependency_analysis = task_dependency_analysis.split('\n\n')
        last_error = None
        cycle_warning = ''  # CRITICAL FIX: Track cycle warnings for retry

        # CRITICAL FIX: Use for-else to avoid "5th attempt success = failure" bug
        # else block only executes if loop completes without break (all 5 attempts failed)
        for attempt in range(1, 6):
            try:
                dependency_DAG = self.dag_construction(tasknum, modeling_problem, problem_analysis, modeling_solution, task_descriptions, task_dependency_analysis, cycle_warning)

                # Try multiple extraction strategies with tiered approach
                dependency_DAG_string = dependency_DAG.strip()

                # Tier 1: Always apply (low risk, high safety)
                dependency_DAG_string = self._apply_tier1_fixes(dependency_DAG_string)

                try:
                    # Try parsing with Tier 1 fixes
                    self.DAG = json.loads(dependency_DAG_string)
                except json.JSONDecodeError as tier1_error:
                    print(f"[DEBUG] Tier 1 fixes failed: {tier1_error.msg}")
                    print(f"[DEBUG] Attempting Tier 2 fixes (medium risk)...")

                    # Tier 2: Apply if Tier 1 fails (medium risk)
                    dependency_DAG_string = self._apply_tier2_fixes(dependency_DAG_string)

                    try:
                        # Try parsing with Tier 2 fixes
                        self.DAG = json.loads(dependency_DAG_string)
                    except json.JSONDecodeError as tier2_error:
                        print(f"[DEBUG] Tier 2 fixes failed: {tier2_error.msg}")
                        print(f"[DEBUG] Attempting Tier 3 fixes (high risk, last resort)...")

                        # Tier 3: Apply only if Tier 2 fails (high risk)
                        dependency_DAG_string = self._apply_tier3_fixes(dependency_DAG_string)

                        # Final attempt with Tier 3 fixes
                        if attempt == 1:
                            print(f"[DEBUG] Attempt {attempt}: JSON preview (first 200 chars): {dependency_DAG_string[:200]}...")
                            print(f"[DEBUG] Total JSON length: {len(dependency_DAG_string)} chars")

                        # Try parsing with better error handling
                        self.DAG = json.loads(dependency_DAG_string)

                # CRITICAL FIX: Normalize DAG to use string keys for consistency
                # LLM may generate integer keys or string keys, so we normalize to strings
                # This prevents KeyError when mathematical_modeling.py accesses coordinator.DAG[str(task_id)]
                normalized_DAG = {}
                for node, dependencies in self.DAG.items():
                    node_str = str(node)
                    deps_str = [str(dep) for dep in dependencies]
                    normalized_DAG[node_str] = deps_str
                self.DAG = normalized_DAG

                print(f"[OK] DAG JSON parsed and normalized successfully on attempt {attempt}")

                # CRITICAL FIX: Now check for cycles BEFORE breaking
                # This was previously AFTER the retry loop, so cycle errors weren't caught
                order = self.compute_dag_order(self.DAG)
                print(f"[OK] DAG validated (acyclic) on attempt {attempt}")
                return order  # Success: return immediately
            except ValueError as e:
                # CRITICAL FIX: Detect cycle errors specifically
                if "Graph contains a cycle" in str(e):
                    print(f"[WARNING] Attempt {attempt}/5: DAG contains cycle - retrying with explicit cycle prevention")
                    last_error = e
                    # Build cycle warning for next attempt
                    cycle_warning = """PREVIOUS ATTEMPT FAILED: The dependency graph you generated contained cycles (circular dependencies).

Example of INVALID cyclic dependencies:
- Task 1 depends on Task 2
- Task 2 depends on Task 1

REQUIREMENTS:
1. You MUST create an ACYCLIC graph (Directed Acyclic Graph - DAG)
2. No task should directly or indirectly depend on itself
3. Dependencies should only flow FROM earlier tasks TO later tasks
4. If Task A depends on Task B, then Task B cannot depend on Task A
5. The final execution order must be a valid topological sort

VALID example:
{
  "1": [],
  "2": ["1"],
  "3": ["1", "2"],
  "4": ["3"]
}

Think carefully about the logical flow of the problem and ensure dependencies are one-way only."""
                    continue
                else:
                    # Re-raise non-cycle ValueError
                    raise
            except Exception as e:
                last_error = e
                print(f"[WARNING] DAG parsing attempt {attempt} failed: {e}")
                print(f"[DEBUG] LLM response preview: {dependency_DAG[:300]}...")
                continue
        else:
            # CRITICAL FIX: for-else means all 5 attempts failed
            print(f"[ERROR] All 5 DAG parsing attempts failed")
            raise last_error or Exception("DAG parsing failed after 5 attempts")
