"""
Academic Paper Generator

Generates academic papers in LaTeX format from structured JSON data using
language models to create content for each section.
"""

import json
import subprocess
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from string import Template  # Required for template substitution


# --------------------------------
# CRITICAL FIX: Safe template substitution
# --------------------------------
#
# This project uses `string.Template` to avoid Python `.format()` conflicts
# with LaTeX/JSON braces. However, LaTeX and prompt text often include `$`
# (math mode, currency, etc.). In `string.Template`, a single `$` starts a
# placeholder and can raise `ValueError: Invalid placeholder`.
#
# To make prompts robust, we:
# 1) Temporarily protect real placeholders ($chapter_path, etc.)
# 2) Escape all remaining `$` as `$$` (literal $ in Template)
# 3) Restore placeholders
# 4) Use `safe_substitute` to avoid KeyError on missing fields

_TPL_SENTINEL_PREFIX = "@@TPL_"
_TPL_SENTINEL_SUFFIX = "_END@@"


def safe_template_substitute(template_str: str, **kwargs: str) -> str:
    """Template substitution that is safe with LaTeX '$' and missing keys."""
    if template_str is None:
        template_str = ""

    # Step 1: Protect intended placeholders.
    protected = template_str
    for key in kwargs.keys():
        protected = protected.replace(f"${key}", f"{_TPL_SENTINEL_PREFIX}{key}{_TPL_SENTINEL_SUFFIX}")

    # Step 2: Escape all remaining '$' so Template won't interpret them.
    protected = protected.replace("$", "$$")

    # Step 3: Restore placeholders.
    for key in kwargs.keys():
        protected = protected.replace(f"{_TPL_SENTINEL_PREFIX}{key}{_TPL_SENTINEL_SUFFIX}", f"${key}")

    # Step 4: Substitute.
    return Template(protected).safe_substitute(**kwargs)


# --------------------------------
# CRITICAL FIX: JSON Sanitization for Stage 4
# --------------------------------

def escape_latex_special_chars(text: str) -> str:
    """
    [CORE DEFENSE FUNCTION] Convert plain text to LaTeX-safe text.
    Covers ALL LaTeX special characters, not just braces.

    This prevents LaTeX compilation errors from characters like:
    - % (comment character)
    - $ (math mode)
    - & (table separator)
    - # (parameter)
    - _ (subscript)
    - ^ (superscript)
    - ~ (non-breakable space)
    - \ (line break)
    - { } (grouping)
    """
    if not isinstance(text, str):
        return str(text)

    # Remove markdown code blocks first (they don't belong in LaTeX)
    text = re.sub(r'```[a-zA-Z]*', '', text)
    text = text.replace('```', '')

    # Complete LaTeX escape mapping
    latex_escapes = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }

    # Use regex for single-pass replacement (more efficient, prevents recursive replacement)
    import re
    regex = re.compile('|'.join(re.escape(key) for key in latex_escapes.keys()))

    def replace_match(match):
        return latex_escapes[match.group()]

    return regex.sub(replace_match, text)


def sanitize_json_for_latex(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    CRITICAL FIX: Sanitize JSON data to prevent LaTeX compilation errors.

    This is the LAST LINE OF DEFENSE before PDF generation.
    Recursively traverses JSON structure and escapes ALL LaTeX special characters.

    Common issues from LLM-generated JSON:
    1. Unescaped special characters: "Accuracy & Precision were > 95% due to parameter_tuning"
    2. Markdown code blocks in values
    3. Invalid UTF-8 characters
    4. Missing required fields
    """
    def sanitize_value(value):
        if isinstance(value, str):
            # Remove markdown code blocks
            value = re.sub(r'```[a-zA-Z]*', '', value)
            value = value.replace('```', '')

            # Execute FULL character escaping
            return escape_latex_special_chars(value.strip())

        elif isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [sanitize_value(v) for v in value]

        else:
            return value

    try:
        # Deep copy to avoid modifying original runtime JSON object
        import copy
        safe_data = copy.deepcopy(json_data)
        return sanitize_value(safe_data)

    except Exception as e:
        print(f"[ERROR] JSON sanitization failed: {e}")
        # If sanitization fails, return original to avoid breaking pipeline
        # Note: This may cause LaTeX compilation to fail, but at least preserves JSON
        return json_data


# --------------------------------
# A1. HOTFIX: Escape braces for .format() calls
# --------------------------------

def escape_braces_for_format(s: str) -> str:
    """
    HOTFIX: Escape curly braces in strings to prevent ValueError: unexpected '{' in field name.

    CRITICAL: Any string passed to .format() containing { or } must be escaped.
    This applies to:
    - JSON content (which contains many braces)
    - LaTeX content (which uses { for commands)
    - User-generated text

    Args:
        s: Input string that may contain unescaped braces

    Returns:
        String with all { replaced by {{ and } replaced by }}
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        return str(s)
    return s.replace("{", "{{").replace("}", "}}")


def validate_solution_json(json_data: Dict[str, Any]) -> List[str]:
    """
    Validate solution JSON structure before LaTeX generation.

    Returns list of errors (empty if valid).
    """
    errors = []

    # Check required top-level fields
    required_fields = ['tasks']
    for field in required_fields:
        if field not in json_data:
            errors.append(f"Missing required field: {field}")

    # Validate each task
    if 'tasks' in json_data:
        for i, task in enumerate(json_data['tasks']):
            task_errors = validate_task_result(task)
            for error in task_errors:
                errors.append(f"Task {i+1}: {error}")

    return errors


def clean_llm_json_output(output_text: str) -> Dict[str, Any]:
    """
    CRITICAL FIX: Clean LLM-generated JSON before parsing.

    Stage 4 often fails with "unexpected '{' in field name" because
    LLM generates malformed JSON with:
    - Unescaped braces
    - Markdown code blocks
    - Trailing commas
    - Comments

    This function pre-processes the output before JSON parsing.
    """
    if not output_text:
        raise ValueError("Empty output from LLM")

    # Step 1: Remove markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    output_text = re.sub(r'```(?:json)?\s*\n?', '', output_text)
    output_text = re.sub(r'```\s*$', '', output_text)

    # Step 2: Remove comments (// or #)
    lines = []
    for line in output_text.split('\n'):
        # Remove // comments
        if '//' in line:
            line = line[:line.index('//')]
        # Remove # comments (if not in string)
        if '#' in line and '"' not in line:
            line = line[:line.index('#')]
        lines.append(line)
    output_text = '\n'.join(lines)

    # Step 3: Remove trailing commas
    # Pattern: "key": value,  }  →  "key": value   }
    output_text = re.sub(r',\s*([}\]])', r'\1', output_text)

    # Step 4: Now parse with improved parser
    try:
        return parse_llm_output_to_json(output_text)
    except Exception as e:
        # Last resort: try to extract JSON using regex
        print(f"[WARN] Standard parsing failed: {e}, attempting fallback extraction")
        return parse_llm_output_to_json(output_text)

# Import statements would be here in a real application
from prompt.template import PAPER_CHAPTER_PROMPT, PAPER_CHAPTER_WITH_PRECEDING_PROMPT, PAPER_INFO_PROMPT, PAPER_NOTATION_PROMPT
from llm.llm import LLM
from utils.utils import parse_llm_output_to_json
from utils.data_models import validate_task_result, normalize_task_dict

# --------------------------------
# Data Models
# --------------------------------

@dataclass
class Chapter:
    """Represents a chapter in the paper with its hierarchical structure and content."""
    path: List[str]  # Hierarchical path (e.g., ["Problem Analysis", "Task 1 Analysis"])
    content: str = ""
    title: str = ""
    is_generated: bool = False
    needs_content: bool = False
    
    @property
    def path_string(self) -> str:
        """Returns the full path as a string (e.g., 'Problem Analysis > Task 1 Analysis')"""
        return " > ".join(self.path)
    
    @property
    def depth(self) -> int:
        """Returns the heading level (depth in hierarchy)"""
        return len(self.path)
    
    @property
    def display_title(self) -> str:
        """Returns the chapter title to display (custom title or last path element)"""
        return self.title if self.title else self.path[-1]

# --------------------------------
# Language Model Interface
# --------------------------------

def escape_underscores_in_quotes(text):
    pattern = r'(".*?")|(\'.*?\')'
    def replace_underscores(match):
        content = match.group(0)[1:-1]
        escaped_content = content.replace('_', r'\_')
        return f'"{escaped_content}"' if match.group(0).startswith('"') else f"'{escaped_content}'"
    
    result = re.sub(pattern, replace_underscores, text, flags=re.DOTALL)
    return result


class ContentGenerator:
    """Interface for generating content using language models"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_chapter_content(self, prompt: str) -> Dict[str, str]:
        """Generate chapter content using the language model"""
        response = self.llm.generate(prompt)
        response = escape_underscores_in_quotes(response)
        response = response.replace("```latex", "").replace("```", "")
        # return self._parse_latex_response(response)
        return response
    
    def _parse_latex_response(self, latex_string: str) -> Dict[str, str]:
        """Parse LLM response from LaTeX format"""
        pattern = r"```latex\s*\\chapter{\s*(.*?)\s*}\s*(.*)```"
        match = re.match(pattern, latex_string.strip(), re.DOTALL)
        
        if match:
            return {
                "title": match.group(1).strip(),
                "content": match.group(2).strip()
            }
        
        # Fallback if format doesn't match
        return {
            "title": "",
            "content": latex_string
        }

# --------------------------------
# Paper Structure
# --------------------------------

class OutlineGenerator:
    """Creates the hierarchical structure of the paper"""
    
    def create_outline(self, task_count: int) -> List[Chapter]:
        """Create a complete chapter structure based on number of tasks"""
        print(f"Creating paper outline for {task_count} tasks")
        
        # Define the structure template
        outline = self._create_base_outline(task_count)
        
        # Create chapter objects
        chapters = []
        for path in outline:
            # A chapter needs content if it's a leaf node (has no children)
            needs_content = not any(other[:len(path)] == path and len(other) > len(path) 
                                   for other in outline)
            chapters.append(Chapter(path=path, needs_content=needs_content))
        
        content_chapters = sum(1 for c in chapters if c.needs_content)
        print(f"Created {len(chapters)} sections, {content_chapters} require content generation")
        for chapter in chapters:
            print(chapter.path_string)
        return chapters
    
    def _create_base_outline(self, task_count: int) -> List[List[str]]:
        """Define the hierarchical structure of the paper"""
        # Define the template structure
        outline = [
            ["Problem Restatement", "Problem Background"],
            ["Problem Restatement", "Problem Statement"],
            ["Model Assumptions"],
            ["Explanation of Assumptions"],
            ["Problem Analysis"]
        ]
        
        # Add task-specific analysis chapters
        for i in range(1, task_count + 1):
            outline.append(["Problem Analysis", f"Task {i} Analysis"])
        
        outline.append(["Solution to the Problem"])
        
        # Add task-specific solution chapters
        for i in range(1, task_count + 1):
            outline.append(["Solution to the Problem", f"Task {i} Solution", "Model Setup: Assumptions and Chain Models"])
            outline.append(["Solution to the Problem", f"Task {i} Solution", "Model Calculation"])
        
        # Add conclusion and reference sections
        outline.extend([
            ["Model Conclusion", "Model Advantages"],
            ["Model Conclusion", "Model Limitations"],
            ["Notation and Explanations"]  
        ])
        
        return outline

    def generate_chapter_relevance_map(self, task_count: int) -> Dict[str, List[str]]:
        """
        Dynamically generate chapter relevance mapping based on the number of tasks.
        
        Args:
            task_count: Number of tasks in the paper
            
        Returns:
            Dictionary mapping chapter paths to lists of related chapter paths
        """
        relevance_map = {}

        for i in range(1, task_count + 1):
            setup_path = f"Solution to the Problem > Task {i} Solution > Model Setup: Assumptions and Chain Models"
            relevance_map[setup_path] = [f"Problem Analysis > Task {i} Analysis"]

        for i in range(1, task_count + 1):
            calculation_path = f"Solution to the Problem > Task {i} Solution > Model Calculation"
            relevance_map[calculation_path] = [
                f"Problem Analysis > Task {i} Analysis",
                f"Solution to the Problem > Task {i} Solution > Model Setup: Assumptions and Chain Models",
            ]
        
        # Model conclusion chapters should include all task solutions
        task_solutions = []
        for i in range(1, task_count + 1):
            task_solutions += [
                f"Solution to the Problem > Task {i} Solution > Model Calculation",
                f"Solution to the Problem > Task {i} Solution > Model Setup: Assumptions and Chain Models"
            ]
        
        relevance_map["Model Conclusion > Model Advantages"] = task_solutions.copy()
        relevance_map["Model Conclusion > Model Limitations"] = task_solutions.copy()
        relevance_map["Notation and Explanations"] = task_solutions.copy()
        
        return relevance_map


# --------------------------------
# Context Extraction
# --------------------------------

class ContextExtractor:
    """Extracts relevant data from JSON for each chapter"""
    
    def get_context_for_chapter(self, chapter: Chapter, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant JSON data for a specific chapter"""
        path = chapter.path
        
        # Handle different chapter types
        if path == ["Problem Restatement", "Problem Background"]:
            return {"problem_background": data.get("problem_background", "")}
            
        elif path == ["Problem Restatement", "Problem Statement"]:
            return {"problem_requirement": data.get("problem_requirement", "")}
            
        elif path == ["Model Assumptions"]:
            return self._get_assumptions_context(data)
    
        elif path == ["Explanation of Assumptions"]:
            return {}
            
        elif self._is_task_analysis(path):
            return self._get_task_analysis_context(path, data)
            
        elif self._is_model_setup(path):
            return self._get_model_setup_context(path, data)
            
        elif self._is_model_calculation(path):
            return self._get_model_calculation_context(path, data)
            
        # Default empty context for other sections
        return {}
    
    def _get_assumptions_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for assumptions sections"""
        context = {"problem_analysis": data.get("problem_analysis", "")}
        
        # Extract task modeling information
        keys = ['task_description', 'task_analysis', 'mathematical_modeling_process']
        context["tasks"] = [
            {k: v for k, v in task.items() if k in keys}
            for task in data['tasks']
        ]
        
        return context
    
    def _get_task_analysis_context(self, path: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for task analysis sections"""
        task_idx = self._extract_task_index(path[1])
        if not self._is_valid_task_index(task_idx, data):
            return {}

        task_data = data["tasks"][task_idx]
        keys = ['task_analysis', 'task_description']
        return {
            f'task_{task_idx+1}': {
                k: v for k, v in task_data.items() if k in keys
            }
        }

    def _get_model_setup_context(self, path: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for model setup sections"""
        task_idx = self._extract_task_index(path[1])
        if not self._is_valid_task_index(task_idx, data):
            return {}

        task_data = data["tasks"][task_idx]
        # CRITICAL FIX: computational_solving.py writes 'modeling_formulas', not 'preliminary_formulas'
        # Changed from: keys = ['preliminary_formulas', 'mathematical_modeling_process']
        keys = ['modeling_formulas', 'mathematical_modeling_process']
        return {
            f'task_{task_idx+1}': {
                k: task_data.get(k, "") for k in keys
            }
        }

    def _get_model_calculation_context(self, path: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Get context for model calculation sections"""
        task_idx = self._extract_task_index(path[1])
        if not self._is_valid_task_index(task_idx, data):
            return {}

        task_data = data["tasks"][task_idx]
        keys = ['mathematical_modeling_process', 'execution_result', 'solution_interpretation', 'subtask_outcome_analysis']
        return {
            f'task_{task_idx+1}': {
                k: task_data.get(k, "") for k in keys
            }
        }
    
    def _is_task_analysis(self, path: List[str]) -> bool:
        """Check if path is a task analysis section"""
        return (len(path) == 2 and 
                path[0] == "Problem Analysis" and 
                path[1].startswith("Task "))
    
    def _is_model_setup(self, path: List[str]) -> bool:
        """Check if path is a model setup section"""
        return (len(path) == 3 and 
                path[0] == "Solution to the Problem" and 
                path[1].startswith("Task ") and 
                path[2] == "Model Setup: Assumptions and Chain Models")
    
    def _is_model_calculation(self, path: List[str]) -> bool:
        """Check if path is a model calculation section"""
        return (len(path) == 3 and 
                path[0] == "Solution to the Problem" and 
                path[1].startswith("Task ") and 
                path[2] == "Model Calculation")

    def _extract_task_index(self, task_string: str) -> int:
        """Extract task index from strings like 'Task 1 Analysis'"""
        try:
            return int(task_string.split()[1]) - 1  # Convert to 0-indexed
        except (IndexError, ValueError):
            return -1

    def _is_valid_task_index(self, index: int, data: Dict[str, Any]) -> bool:
        """Check if the task index is valid"""
        return 0 <= index < len(data.get("tasks", []))

# --------------------------------
# Prompt Creation
# --------------------------------

class PromptCreator:
    """Creates prompts for the language model"""
    
    def __init__(self):
        pass
    
    def create_prompt(self, 
                     chapter: Chapter, 
                     context: Dict[str, Any], 
                     previous_chapters: List[Chapter]) -> str:
        """Create a prompt for generating chapter content"""
        # Format JSON context
        json_str = json.dumps(context, indent=2)
        
        # Format previous chapters
        previous_text = self._format_previous_chapters(previous_chapters)

        if chapter.path == ["Notation and Explanations"]:
            # Use safe template substitution to avoid '$' placeholder errors
            return safe_template_substitute(
                PAPER_NOTATION_PROMPT,
                previous_chapters=previous_text,
            )
        else:
            if json_str == '{}':
                return safe_template_substitute(
                    PAPER_CHAPTER_WITH_PRECEDING_PROMPT,
                    chapter_path=chapter.path_string,
                    previous_chapters=previous_text
                )
            else:
                # Build the prompt using the template
                # Use safe template substitution to avoid '$' placeholder errors
                return safe_template_substitute(
                    PAPER_CHAPTER_PROMPT,
                    chapter_path=chapter.path_string,
                    json_context=json_str,
                    previous_chapters=previous_text
                )
    
    def _format_previous_chapters(self, previous_chapters: List[Chapter]) -> str:
        """Format previously completed chapters for context"""
        if not previous_chapters:
            return ""
            
        text = ""
        for chapter in previous_chapters:
            text += f"Chapter: {chapter.path_string}\n"
            # text += f"Title: {chapter.display_title}\n"
            text += f"{chapter.content}\n\n"
        return text


# --------------------------------
# Document Assembly
# --------------------------------

class LatexDocumentAssembler:
    """Assembles the final LaTeX document from generated chapters"""
    
    def create_document(self, chapters: List[Chapter], metadata: Dict[str, Any]) -> str:
        """Create a complete LaTeX document"""
        # Reorder chapters (move Notation chapter after Explanation of Assumptions)
        ordered_chapters = self._reorder_chapters(chapters)
        
        # Build document parts
        document_parts = [
            self._create_preamble(metadata),
            self._create_abstract(metadata),
            "\\maketitle",
            "\\renewcommand\\cfttoctitlefont{\\hfil\\Large\\bfseries}",
            "\\tableofcontents",
            "\\newpage",
            self._create_body(ordered_chapters, metadata),
            "\\end{document}"
        ]
        
        return "\n\n".join(document_parts)
    
    def _reorder_chapters(self, chapters: List[Chapter]) -> List[Chapter]:
        """Reorder chapters for better document structure"""
        reordered = []
        notation_chapter = next((ch for ch in chapters if ch.path == ["Notation and Explanations"]), None)
        
        for chapter in chapters:
            if chapter.path != ["Notation and Explanations"]:
                reordered.append(chapter)
                # Insert notation chapter after Explanation of Assumptions
                if notation_chapter and chapter.path == ["Explanation of Assumptions"]:
                    reordered.append(notation_chapter)
                    
        return reordered
    
    def _add_figure(self, figures: List[str], latex_dir: Optional[str] = None) -> str:
        """
        Add a figure to the content.

        CRITICAL FIX: Handle missing/failed chart files gracefully.
        If a figure file doesn't exist, use a placeholder instead of failing the entire LaTeX compilation.

        CRITICAL FIX 2: Use latex_dir for existence checks (relative paths need resolution).
        """
        figure_str: List[str] = []
        base_dir = latex_dir or '.'

        for i, figure_path in enumerate(figures):
            # Resolve absolute path for existence check
            abs_path = figure_path
            if not os.path.isabs(figure_path):
                abs_path = os.path.join(base_dir, figure_path)

            if not os.path.exists(abs_path):
                # Figure file missing - use placeholder
                name = os.path.splitext(os.path.basename(figure_path))[0].replace('_', '\\_')
                print(f"[WARN] Figure file not found: {figure_path} (resolved: {abs_path}) - using placeholder")
                figure_str.append(f"""
\\begin{{figure}}[H]
\\centering
\\fbox{{\\begin{{minipage}}{{0.8\\textwidth}}
\\centering
\\vspace{{1cm}}
\\textbf{{Figure Not Available}}
\\par
\\vspace{{0.5cm}}
\\textit{{The chart for ``{name}'' could not be generated.}}
\\par
\\vspace{{1cm}}
\\end{{minipage}}}}
\\caption{{{name} (chart generation failed)}}
\\end{{figure}}
""")
                continue

            # Figure file exists - include it normally
            # CRITICAL: Use POSIX path (forward slashes) for LaTeX
            latex_path = figure_path.replace('\\\\', '/').replace('\\', '/')
            name = os.path.splitext(os.path.basename(figure_path))[0].replace('_', '\\_')
            figure_str.append(f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.5\\textwidth]{{{latex_path}}}
\\caption{{{name}}}
\\end{{figure}}
""")

        # CRITICAL: If no figures at all, add a note
        if not figure_str:
            print("[INFO] No figures available for LaTeX document")
            figure_str.append("""
% No figures were successfully generated
% This is expected if chart generation failed
""")

        return figure_str


    def _add_code(self, codes: List[str]) -> str:
        r"""
\subsection*{Python Code}
\subsubsection*{main1.py}

\begin{lstlisting}[language=Python, frame=single, basicstyle=\ttfamily\small]
def main1():
    pass
\end{lstlisting}
        """
        code_str = [
            "\\clearpage",
            "\\section{Appendix}",
        ]
        for i, code_path in enumerate(codes):
            with open(code_path, 'r') as f:
                code = f.read()
            name = code_path.split('/')[-1].replace('_', '\\_')
            code_str.append(f"""
\\subsubsection*{{{name}}}

\\begin{{lstlisting}}[language=Python, frame=single, basicstyle=\\ttfamily\\small]
{code}
\\end{{lstlisting}}
""")
        return code_str

    def _create_preamble(self, metadata: Dict[str, Any]) -> str:
        """Create LaTeX preamble with document setup"""
        title = metadata.get("title", "paper_title")
        team = metadata.get("team", "team")
        year = metadata.get("year", "2024")
        problem_type = metadata.get("problem_type", "problem_type")
        
        return f"""\\documentclass{{mcmthesis}}
\\mcmsetup{{CTeX = false,
        tcn = {team}, problem = {problem_type},
        year = {year},
        sheet = true, titleinsheet = true, keywordsinsheet = true,
        titlepage = false, abstract = true}}

\\usepackage{{palatino}}
\\usepackage{{algorithm}}
\\usepackage{{algpseudocode}}
\\usepackage{{tocloft}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{array}}
\\usepackage{{tabularx}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{rotating}}
\\usepackage{{longtable}}
\\usepackage{{adjustbox}}
% CRITICAL FIX: Add missing packages needed for LaTeX compilation
\\usepackage{{float}}     % For [H] placement specifier
\\usepackage{{listings}}  % For \\lstlisting code environment
\\usepackage{{xcolor}}    % For listings coloring

\\usepackage{{lastpage}}
\\renewcommand{{\\cftdot}}{{.}}
\\renewcommand{{\\cftsecleader}}{{\\cftdotfill{{\\cftdotsep}}}}
\\renewcommand{{\\cftsubsecleader}}{{\\cftdotfill{{\\cftdotsep}}}}
\\renewcommand{{\\cftsubsubsecleader}}{{\\cftdotfill{{\\cftdotsep}}}}
\\renewcommand{{\\headset}}{{{year}\\\\MCM/ICM\\\\Summary Sheet}}
\\title{{{title}}}

\\begin{{document}}"""
    
    def _create_abstract(self, metadata: Dict[str, str]) -> str:
        """Create the abstract section"""
        return f"""\\begin{{abstract}}
{metadata.get('summary', '')}

\\begin{{keywords}}
{metadata.get('keywords', '')}
\\end{{keywords}}
\\end{{abstract}}"""
    
    def _create_body(self, chapters: List[Chapter], metadata: Dict[str, Any]) -> str:
        """Create the main body of the document from chapters"""
        body_parts = []
        current_path = []
        
        for chapter in chapters:
            # Add section headings
            if chapter.path == ["Model Conclusion", "Model Advantages"] and metadata.get('figures', []):
                body_parts += self._add_figure(metadata['figures'], latex_dir=metadata.get('_latex_dir'))

            for i, section in enumerate(chapter.path):
                # If this path level is new or different
                if i >= len(current_path) or section != current_path[i]:
                    # Update current path
                    if len(current_path) <= i:
                        current_path.append(section)
                    else:
                        current_path[i] = section
                        current_path = current_path[:i+1]  # Truncate the path
                
                    # Use custom title if available for the last level
                    title = chapter.display_title if i == chapter.depth - 1 else section
                    
                    # Add section heading at appropriate level
                    if i == 0:
                        body_parts.append(f"\\section{{{title}}}")
                    elif i == 1:
                        body_parts.append(f"\\subsection{{{title}}}")
                    elif i == 2:
                        body_parts.append(f"\\subsubsection{{{title}}}")
            
            # Add chapter content if generated
            if chapter.is_generated and chapter.content:
                body_parts.append(chapter.content)

        body_parts.append("\\section{References}")
        body_parts += self._add_code(metadata['codes'])
        return "\n\n".join(body_parts)

# --------------------------------
# File Operations
# --------------------------------

class FileManager:
    """Handles file operations for saving papers and generating PDFs"""
    
    @staticmethod
    def save_to_file(content: str, filepath: str) -> None:
        """Save content to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Document saved to {filepath}")

    @staticmethod
    def extract_latex_errors(log_output: str) -> list:
        """
        Extract LaTeX error messages from pdflatex output.

        Args:
            log_output: Full pdflatex stdout/stderr

        Returns:
            List of error dictionaries
        """
        errors = []
        lines = log_output.split('\n')

        for i, line in enumerate(lines):
            if line.strip().startswith('!'):
                error_msg = line.strip()
                context = []
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('!'):
                        context.append(lines[j].strip())
                    elif lines[j].startswith('!'):
                        break
                errors.append({
                    'error': error_msg,
                    'context': '\n'.join(context)
                })

        return errors

    @staticmethod
    def fix_latex_with_llm(latex_content: str, errors: list, llm) -> str:
        """
        Use LLM to fix LaTeX errors.

        Args:
            latex_content: Original LaTeX source
            errors: List of error dictionaries
            llm: LLM instance

        Returns:
            Fixed LaTeX source code
        """
        error_text = "\n\n".join([
            f"Error {i+1}: {err['error']}\nContext: {err['context']}"
            for i, err in enumerate(errors[:10])  # Limit to first 10 errors
        ])

        prompt = f"""Fix the following LaTeX compilation errors. Return ONLY the fixed LaTeX code.

## Errors:
{error_text}

## Current LaTeX:
```latex
{latex_content}
```

## Instructions:
1. Fix ONLY the errors mentioned above
2. Keep everything else unchanged
3. Return complete LaTeX code starting with \\documentclass
4. End with \\end{{document}}
5. NO markdown code blocks, NO explanations
6. Ensure all environments are properly closed

Fixed LaTeX:"""

        try:
            response = llm.generate(prompt)
            response = response.replace("```latex", "").replace("```", "")
            return response.strip()
        except Exception as e:
            print(f"[ERROR] LLM fixing failed: {e}")
            return latex_content

    @staticmethod
    def _print_compilation_errors(result, latex_path: str = None) -> None:
        """
        Print detailed compilation errors from subprocess result (P2-2 FIX: Enhanced).

        Args:
            result: subprocess.CompletedProcess result
            latex_path: Optional path to .tex file for locating .log file
        """
        print("="*70)
        print("LATEX COMPILATION ERROR DETAILS")
        print("="*70)

        # Extract and display critical errors (lines starting with '!')
        all_output = result.stdout + result.stderr
        error_lines = []
        context_before = {}

        lines = all_output.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('!'):
                error_lines.append(line.strip())
                # Collect context (2 lines before and after)
                context_start = max(0, i-2)
                context_end = min(len(lines), i+3)
                context_before[line.strip()[:50]] = lines[context_start:context_end]

        if error_lines:
            print(f"\n[CRITICAL ERRORS] Found {len(error_lines)} critical error(s):")
            print("-"*70)
            for err in error_lines[:20]:  # Show first 20 errors
                print(f"  {err}")
                if err[:50] in context_before:
                    context = context_before[err[:50]]
                    print(f"  Context:")
                    for ctx_line in context:
                        if ctx_line and not ctx_line.strip().startswith('!'):
                            print(f"    {ctx_line}")
            if len(error_lines) > 20:
                print(f"  ... and {len(error_lines) - 20} more errors")

        # Print last 1000 chars of stdout (fuller context)
        if result.stdout:
            output_len = len(result.stdout)
            if output_len > 1000:
                print(f"\n[STDOUT] Last 1000 chars of {output_len} total:")
                print("-"*70)
                print(result.stdout[-1000:])
            else:
                print(f"\n[STDOUT] Full output ({output_len} chars):")
                print("-"*70)
                print(result.stdout)

        # Print stderr (usually contains pdflatex error messages)
        if result.stderr:
            err_len = len(result.stderr)
            if err_len > 1000:
                print(f"\n[STDERR] Last 1000 chars of {err_len} total:")
                print("-"*70)
                print(result.stderr[-1000:])
            else:
                print(f"\n[STDERR] Full error output ({err_len} chars):")
                print("-"*70)
                print(result.stderr)

        # P2-2 FIX: Log .log file path for manual inspection
        if latex_path:
            log_path = latex_path.replace('.tex', '.log')
            print(f"\n[INFO] Full LaTeX log file: {log_path}")
            print(f"[INFO] To view errors manually: tail -100 {log_path}")

        print("="*70)
        print("[END ERROR DETAILS]")
        print("="*70)

    @staticmethod
    def generate_pdf(latex_path: str, llm=None, max_iterations: int = 20) -> bool:
        """
        Generate a PDF from a LaTeX file with automatic error fixing (self-iteration).

        Args:
            latex_path: Path to .tex file
            llm: LLM instance for error fixing (optional, enables self-iteration)
            max_iterations: Maximum compilation iterations (default: 20)

        Returns:
            bool: True if PDF generation succeeded, False otherwise
        """
        print(f"Generating PDF from {latex_path}...")

        # Check if LaTeX file exists
        if not os.path.exists(latex_path):
            print(f"[WARNING] LaTeX file not found: {latex_path}")
            print(f"[INFO] PDF generation skipped - LaTeX source file missing")
            return False

        # Read original LaTeX content
        with open(latex_path, 'r', encoding='utf-8') as f:
            original_latex = f.read()

        latex_dir = os.path.dirname(latex_path)
        current_latex = original_latex

        try:
            # Check if pdflatex is available
            import shutil
            if not shutil.which("pdflatex"):
                print(f"[WARNING] pdflatex not found in PATH")
                print(f"[INFO] PDF generation skipped - LaTeX compiler not installed")
                print(f"[INFO] LaTeX file saved at: {latex_path}")
                print(f"[INFO] To generate PDF manually, install LaTeX (e.g., TeX Live, MiKTeX)")
                return False

            # Enable self-iteration if LLM is provided
            if llm is not None:
                print(f"[INFO] Self-iteration enabled (max {max_iterations} attempts)")

            # Self-iteration loop
            for iteration in range(1, max_iterations + 1):
                if llm is not None and iteration > 1:
                    print(f"\n[PDF Generation - Iteration {iteration}/{max_iterations}]")

                # Write current LaTeX content
                with open(latex_path, 'w', encoding='utf-8') as f:
                    f.write(current_latex)

                # Run pdflatex first pass
                if iteration == 1:
                    print("[INFO] Running pdflatex (first pass)...")
                tex_name = os.path.basename(latex_path)
                result1 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=latex_dir,
                )

                # Check if PDF was created successfully
                # IMPORTANT: PDF is created in latex_dir, not in the original path
                pdf_path = os.path.join(latex_dir, os.path.splitext(tex_name)[0] + '.pdf')

                if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                    # PDF created successfully - run second pass for references
                    if iteration == 1:
                        print("[OK] First pass successful")
                    print("[INFO] Running pdflatex (second pass for references)...")

                    result2 = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", tex_name],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=latex_dir,
                    )

                    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                        # Clean up and return success
                        FileManager._clean_temp_files(latex_path)
                        pdf_size = os.path.getsize(pdf_path)
                        print(f"[OK] PDF generated successfully: {pdf_path} ({pdf_size:,} bytes)")
                        if llm is not None:
                            print(f"[INFO] Total iterations: {iteration}")
                        return True

                # Compilation failed - check if we should retry with LLM fixing
                if llm is None:
                    print("[WARNING] Compilation failed and no LLM provided")
                    FileManager._print_compilation_errors(result1, latex_path)  # P2-2 FIX: Pass latex_path
                    return False

                # Extract errors
                combined_output = result1.stdout + result1.stderr
                errors = FileManager.extract_latex_errors(combined_output)

                if not errors:
                    print("[WARNING] No clear errors found in output")
                    FileManager._print_compilation_errors(result1, latex_path)  # P2-2 FIX: Pass latex_path
                    return False

                print(f"[INFO] Found {len(errors)} error(s), attempting LLM fix...")

                # Show first few errors
                for i, err in enumerate(errors[:5]):
                    print(f"  Error {i+1}: {err['error']}")

                # Use LLM to fix errors
                print("[INFO] Requesting LLM to fix errors...")
                current_latex = FileManager.fix_latex_with_llm(current_latex, errors, llm)

                if current_latex == original_latex:
                    print("[WARNING] LLM did not modify the LaTeX")
                    return False

                print("[OK] LLM provided fixed LaTeX, retrying...")

            # Max iterations reached
            print(f"\n[FAILED] Could not generate PDF after {max_iterations} iterations")
            FileManager._print_compilation_errors(result1, latex_path)  # P2-2 FIX: Pass latex_path
            return False

        except subprocess.TimeoutExpired:
            print(f"[ERROR] pdflatex timeout after 60 seconds")
            print(f"[INFO] PDF generation failed - LaTeX compilation took too long")
            return False
        except FileNotFoundError:
            print(f"[WARNING] pdflatex command not found")
            print(f"[INFO] PDF generation skipped - LaTeX compiler not installed")
            print(f"[INFO] LaTeX file saved at: {latex_path}")
            print(f"[INFO] To generate PDF manually, install LaTeX (e.g., TeX Live, MiKTeX)")
            return False
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] pdflatex failed with return code: {e.returncode}")
            print(f"[INFO] PDF generation failed - LaTeX compilation error")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error during PDF generation: {type(e).__name__}: {e}")
            print(f"[INFO] LaTeX file saved at: {latex_path}")
            return False
    
    @staticmethod
    def _clean_temp_files(latex_path: str) -> None:
        """Clean up temporary files created during PDF generation"""
        for ext in ["aux", "log", "toc", "out"]:
            aux_file = latex_path.replace('.tex', f'.{ext}')
            if os.path.exists(aux_file):
                os.remove(aux_file)

# --------------------------------
# Main Paper Generator
# --------------------------------

class PaperGenerator:
    """Main class that orchestrates the paper generation process"""
    
    def __init__(self, llm):
        self.content_generator = ContentGenerator(llm)
        self.outline_generator = OutlineGenerator()
        self.context_extractor = ContextExtractor()
        self.prompt_creator = PromptCreator()
        self.document_assembler = LatexDocumentAssembler()
        self.file_manager = FileManager()
        self.llm = llm
        
    def generate_paper(self, 
                    json_data: Dict[str, Any], 
                    metadata: Dict[str, Any],
                    output_dir: str,
                    filename: str) -> None:
        """Generate a complete academic paper from JSON data"""
        # 1. Create chapter structure
        task_count = len(json_data.get("tasks", []))
        print(f"Starting paper generation with {task_count} tasks")
        chapters = self.outline_generator.create_outline(task_count)
        
        # Generate chapter relevance map if not provided
        chapter_relevance_map = self.outline_generator.generate_chapter_relevance_map(task_count)
        
        # 2. Generate content for each chapter that needs it
        completed_chapters = []
        for chapter in chapters:
            if chapter.needs_content:
                self._generate_chapter_content(chapter, json_data, completed_chapters, chapter_relevance_map)
                completed_chapters.append(chapter)
        
        # 3. Complete metadata if needed
        complete_metadata = self._complete_metadata(chapters, metadata)
        
        # 4. Assemble the final document
        document = self.document_assembler.create_document(chapters, complete_metadata)
        
        # 5. Save and convert to PDF
        latex_path = f"{output_dir}/{filename}.tex"
        self.file_manager.save_to_file(document, latex_path)
        self.file_manager.generate_pdf(latex_path, llm=self.llm, max_iterations=20)
        
    def _generate_chapter_content(self, 
                            chapter: Chapter, 
                            json_data: Dict[str, Any],
                            completed_chapters: List[Chapter],
                            chapter_relevance_map: Dict[str, List[str]]) -> None:
        """Generate content for a single chapter"""
        print(f"Generating content for: {chapter.path_string}")
        
        # Get relevant context data for this chapter
        context = self.context_extractor.get_context_for_chapter(chapter, json_data)
        
        # Get only the relevant completed chapters for context
        relevant_chapters = self._get_relevant_chapters(chapter, completed_chapters, chapter_relevance_map)
        
        # Create prompt and generate content
        prompt = self.prompt_creator.create_prompt(
            chapter, context, relevant_chapters
        )
        # Generate content
        response = self.content_generator.generate_chapter_content(prompt)
        
        # Update chapter with generated content
        # chapter.content = response['content']
        # chapter.title = self._format_title(chapter, response['title'])
        chapter.content = response
        chapter.title = ''
        chapter.is_generated = True
    
    def _get_relevant_chapters(self, 
                         chapter: Chapter, 
                         completed_chapters: List[Chapter],
                         chapter_relevance_map: Dict[str, List[str]]) -> List[Chapter]:
        """Filter completed chapters to only include those relevant to the current chapter"""
        # Get the path string for the current chapter
        current_path = chapter.path_string
        
        # If this chapter has specific relevant chapters defined in the map
        if current_path in chapter_relevance_map:
            relevant_paths = chapter_relevance_map[current_path]
            # Filter completed chapters to only include those in the relevant paths
            return [ch for ch in completed_chapters 
                    if ch.path_string in relevant_paths]
        
        # Default: return all completed chapters if no specific relevance is defined
        return completed_chapters

    def _format_title(self, chapter: Chapter, generated_title: str) -> str:
        """Format title based on chapter type"""
        # Only use custom titles for certain chapter types
        if (chapter.path[0] == "Problem Analysis" or 
            chapter.path[0] == "Solution to the Problem"):
            return generated_title
        return ''
    
    def _complete_metadata(self, 
                        chapters: List[Chapter], 
                        provided_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Complete paper metadata, generating missing fields if needed"""
        # If we need to generate metadata
        if not all(key in provided_metadata for key in 
                ["title", "summary", "keywords"]):
            print("Generating missing paper metadata...")
            
            # Prepare prompt with chapter contents
            chapters_text = "\n\n".join(
                f"Chapter: {ch.path_string}\n{ch.content}"
                for ch in chapters if ch.is_generated
            )

            # Use safe template substitution to avoid '$' placeholder errors
            prompt = safe_template_substitute(PAPER_INFO_PROMPT, paper_chapters=chapters_text)

            # Retry up to 3 times to get valid metadata
            max_retries = 3
            generated_metadata = {}
            
            for attempt in range(max_retries):
                try:
                    metadata_response = self.llm.generate(prompt)
                    # CRITICAL FIX: Use clean_llm_json_output to handle malformed LLM output
                    # This prevents "unexpected '{' in field name" errors
                    generated_metadata = clean_llm_json_output(metadata_response)
                    if not generated_metadata:
                        raise Exception("No metadata generated")
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_retries - 1:  # If this was the last attempt
                        print("All attempts to generate metadata failed")
            # Merge with provided metadata (provided takes precedence)
            return {**generated_metadata, **provided_metadata}
        
        return provided_metadata

# --------------------------------
# Main Function
# --------------------------------

def generate_paper_from_json(llm, json_data: dict, info: dict, output_dir: str, output_name: str) -> None:
    """Generate a paper from JSON data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generator = PaperGenerator(llm)
    generator.generate_paper(json_data, info, output_dir, output_name)


def generate_paper(llm, output_dir, name):
    from pathlib import Path  # [FIX] 引入 Path 以处理路径逻辑

    metadata = {
        "team": "Agent",
        "year": name.split('_')[0],
        "problem_type": name.split('_')[1]
    }

    # =========================================================================
    # [CRITICAL FIX] 智能路径处理，防止 Workspace 重复嵌套
    # =========================================================================
    from pathlib import Path
    out_path = Path(output_dir).resolve()

    print(f"[DEBUG] Raw output_dir received: {out_path}")

    # 检查 output_dir 是否已经指向了 Workspace 目录（大小写不敏感，兼容Windows）
    if out_path.name.lower() == 'workspace':
        workspace_dir = out_path
        # 如果 output_dir 就是 Workspace，那 output_path 的父级才是根目录 (用于日志等)
        output_path = out_path.parent
    else:
        workspace_dir = out_path / 'Workspace'

    # 基于确认好的 workspace_dir 构建所有子路径
    json_file_path = workspace_dir / 'json' / f"{name}.json"
    code_dir = workspace_dir / 'code'
    charts_dir = workspace_dir / 'charts'
    latex_dir = workspace_dir / 'latex'

    print(f"[INFO] Path resolution:")
    print(f"  - Output Dir: {output_dir}")
    print(f"  - Workspace:  {workspace_dir}")
    print(f"  - JSON Path:  {json_file_path}")
    # =========================================================================

    # Collect chart images with relative paths for LaTeX compilation
    metadata['figures'] = []
    metadata['failed_charts'] = []  # CRITICAL: Track failed charts for LaTeX degradation

    if charts_dir.is_dir():
        for f in charts_dir.iterdir():
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    # Try to make path relative to latex/ directory
                    # CRITICAL: Use as_posix() to ensure forward slashes for LaTeX
                    fig_rel = f.relative_to(latex_dir).as_posix()
                except ValueError:
                    # If charts are not under latex_dir (usually they're in charts/)
                    # Calculate relative path like ../charts/xxx.png
                    try:
                        fig_rel = os.path.relpath(str(f), str(latex_dir)).replace(os.sep, '/')
                    except Exception:
                        # Last resort: use absolute path (higher risk in LaTeX)
                        fig_rel = f.as_posix()
                metadata['figures'].append(fig_rel)

    # CRITICAL FIX: Check for failed charts and log them
    # Read the JSON solution to check which charts failed
    try:
        # 使用修复后的 json_file_path 读取
        if not json_file_path.exists():
             print(f"[ERROR] JSON solution file not found at: {json_file_path}")
             return

        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.loads(f.read())

        # CRITICAL FIX: Validate and normalize JSON before using it
        # This prevents "unexpected '{' in field name" errors
        print("[INFO] Validating JSON structure before LaTeX generation...")

        # Normalize task dictionaries (fix field name inconsistencies)
        if 'tasks' in json_data:
            for i, task in enumerate(json_data['tasks']):
                json_data['tasks'][i] = normalize_task_dict(task)

        # Validate JSON structure
        validation_errors = validate_solution_json(json_data)
        if validation_errors:
            print(f"[WARN] JSON validation found {len(validation_errors)} issues:")
            for error in validation_errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            # Continue anyway - sanitization will fix most issues

        # Sanitize JSON data to prevent LaTeX errors
        json_data = sanitize_json_for_latex(json_data)
        print("[INFO] JSON sanitization complete")

        # Iterate through tasks to find failed charts
        for task_idx, task in enumerate(json_data.get('tasks', [])):
            if 'charts' in task:
                for chart_idx, chart in enumerate(task['charts']):
                    if not chart.get('success', False):
                        error_msg = chart.get('error', 'Unknown error')
                        task_id = task.get('task_description', f'Task {task_idx + 1}')
                        metadata['failed_charts'].append({
                            'task': task_id,
                            'chart_index': chart_idx + 1,
                            'error': error_msg
                        })

        if metadata['failed_charts']:
            print(f"[WARN] {len(metadata['failed_charts'])} chart(s) failed - will use placeholders in LaTeX")
            for failed in metadata['failed_charts']:
                print(f"  - {failed['task']} Chart {failed['chart_index']}: {failed['error'][:50]}...")

    except Exception as e:
        print(f"[WARN] Could not check for failed charts: {e}")
        # Continue without failed chart info - LaTeX will use all available figures
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.loads(f.read())

    json_data['tasks'] = json_data['tasks'][:]

    # [FIX] Generate paper in Workspace/latex (using the correctly resolved path)
    generate_paper_from_json(llm, json_data, metadata, str(latex_dir), 'solution')

