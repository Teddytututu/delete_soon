import sys
from pathlib import Path
import json
import os

# CRITICAL: Configure UTF-8 encoding for Windows terminal
# This prevents UnicodeEncodeError when printing special characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older Python versions
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.llm import LLM
from utils.utils import write_json_file, get_info
from utils.execution_tracker import get_tracker
from utils.auto_evaluation import run_auto_evaluation
import time
import argparse
from utils.problem_analysis import problem_analysis
from utils.mathematical_modeling import mathematical_modeling
from utils.computational_solving import computational_solving
from utils.solution_reporting import generate_paper
from functools import partial


def get_model_base_url(model_name: str, provided_base_url: str = None) -> str:
    """
    Auto-detect base_url based on model name.

    GLM models use Zhipu AI endpoint, others use OpenAI-compatible endpoints.
    """
    if provided_base_url and provided_base_url != "https://api.openai.com/v1":
        # User provided custom base_url, use it
        return provided_base_url

    # Auto-detect based on model name
    if model_name.startswith('glm-'):
        # GLM models (Zhipu AI)
        return "https://open.bigmodel.cn/api/paas/v4"
    elif model_name.startswith('qwen'):
        # Qwen models (Alibaba)
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif model_name.startswith('deepseek'):
        # DeepSeek models
        return os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
    else:
        # Default to OpenAI
        return provided_base_url or "https://api.openai.com/v1"


def run_mmbench_evaluation(key, evaluation_model, solution_path, output_dir, logger_manager):
    """
    Run MMBench evaluation on the generated solution.

    Args:
        key: API key for evaluation model
        evaluation_model: Model name for evaluation (supports GLM models)
        solution_path: Path to solution JSON file
        output_dir: Output directory (evaluation results go to logs/evaluation/)
        logger_manager: Logger instance

    Returns:
        Path to evaluation results directory
    """
    from MMBench.evaluation.model import gpt
    from MMBench.evaluation.prompts import (
        generate_problem_analysis_prompt,
        generate_modeling_rigorousness_prompt,
        generate_practicality_and_scientificity_prompt,
        generate_result_and_bias_analysis_prompt
    )

    main_logger = logger_manager.get_logger('main')

    # Auto-detect base_url based on evaluation model
    base_url = get_model_base_url(evaluation_model, None)
    main_logger.info(f"[INFO] Using evaluation model: {evaluation_model}")
    main_logger.info(f"[INFO] Using evaluation base_url: {base_url}")

    # Create LLM function for evaluation
    llm = partial(gpt, base_url=base_url, key=key, model=evaluation_model, temperature=0.7, max_tokens=4000)

    # Load solution
    main_logger.info("[INFO] Loading solution for evaluation...")
    try:
        with open(solution_path, 'r', encoding='utf-8') as f:
            solution_data = json.load(f)
    except Exception as e:
        main_logger.error(f"[ERROR] Failed to load solution: {e}")
        raise

    # Create evaluation directory in logs/evaluation/
    evaluation_dir = Path(output_dir) / "logs" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    main_logger.info(f"[INFO] Evaluation directory: {evaluation_dir}")

    # Generate evaluation prompts
    main_logger.info("[INFO] Generating evaluation prompts...")
    problem_analysis_prompt = generate_problem_analysis_prompt(solution_data)
    modeling_rigorousness_prompt = generate_modeling_rigorousness_prompt(solution_data)
    practicality_prompt = generate_practicality_and_scientificity_prompt(solution_data)
    result_analysis_prompt = generate_result_and_bias_analysis_prompt(solution_data)

    # Run evaluations
    main_logger.info("[INFO] Running problem analysis evaluation...")
    problem_analysis_results = llm(problem_analysis_prompt)
    main_logger.info("[OK] Problem analysis evaluation complete")

    main_logger.info("[INFO] Running modeling rigorousness evaluation...")
    modeling_results = llm(modeling_rigorousness_prompt)
    main_logger.info("[OK] Modeling rigorousness evaluation complete")

    main_logger.info("[INFO] Running practicality and scientificity evaluation...")
    practicality_results = llm(practicality_prompt)
    main_logger.info("[OK] Practicality and scientificity evaluation complete")

    main_logger.info("[INFO] Running result and bias analysis evaluation...")
    result_results = llm(result_analysis_prompt)
    main_logger.info("[OK] Result and bias analysis evaluation complete")

    # Parse evaluation results
    import re
    def parse_evaluation_data(text):
        reason_pattern = r'<reason>\s*(.*?)\s*</reason>'
        score_pattern = r'<score>\s*(\d+)\s*</score>'
        reasons = re.findall(reason_pattern, text, re.DOTALL)
        scores = re.findall(score_pattern, text, re.DOTALL)
        if not reasons or not scores:
            return {}
        result_dict = {}
        for i, (reason, score) in enumerate(zip(reasons, scores), 1):
            result_dict[f"analysis_{i}"] = {
                "reason": reason.strip(),
                "score": int(score)
            }
        return result_dict

    # Compile all evaluation data
    all_evaluation_data = {
        "problem_id": Path(solution_path).stem,
        "evaluation_model": evaluation_model,
        "analysis_evaluation": parse_evaluation_data(problem_analysis_results[0]),
        "modeling_rigorousness_evaluation": parse_evaluation_data(modeling_results[0]),
        "practicality_and_scientificity_evaluation": parse_evaluation_data(practicality_results[0]),
        "result_and_bias_analysis_evaluation": parse_evaluation_data(result_results[0])
    }

    # Save evaluation results to JSON
    evaluation_json_path = evaluation_dir / "evaluation_results.json"
    with open(evaluation_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_evaluation_data, f, ensure_ascii=False, indent=2)
    main_logger.info(f"[OK] Evaluation results saved to {evaluation_json_path}")

    # Save full evaluation text
    evaluation_text_path = evaluation_dir / "evaluation_results.txt"
    all_text = (
        problem_analysis_results[0] + "\n\n" +
        modeling_results[0] + "\n\n" +
        practicality_results[0] + "\n\n" +
        result_results[0]
    )
    with open(evaluation_text_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    main_logger.info(f"[OK] Evaluation text saved to {evaluation_text_path}")

    return str(evaluation_dir)


def run(key, problem_path, config, name, dataset_path, output_dir, logger_manager,
        enable_latent_summary=False, enable_latent_summary_stages=None,
        enable_summary_cache=True, failure_handler=None):
    """
    Run MM-Agent experiment with comprehensive execution tracking.

    Args:
        key: API key for LLM
        problem_path: Path to problem JSON file
        config: Configuration dictionary
        name: Problem/task name
        dataset_path: Path to dataset directory
        output_dir: Output directory path
        logger_manager: MMExperimentLogger instance for structured logging
        enable_latent_summary: Whether to generate LLM-based summaries for each API call
        enable_latent_summary_stages: List of stages to generate summaries for (None = all)
        enable_summary_cache: Whether to cache and reuse similar summaries
        failure_handler: FailureHandler instance for minimal JSON fallback (STEP 4)

    Returns:
        Solution dictionary
    """
    # Initialize execution tracker FIRST
    tracker = get_tracker(output_dir, logger_manager)
    main_logger = logger_manager.get_logger('main')

    # Initialize LLM with tracker and latent summary option
    llm = LLM(config['model_name'], key, logger_manager=logger_manager, tracker=tracker,
             enable_latent_summary=enable_latent_summary,
             latent_summary_stages=enable_latent_summary_stages,
             enable_summary_cache=enable_summary_cache)

    if enable_latent_summary:
        main_logger.info("[INFO] Latent LLM summaries ENABLED")
        if enable_latent_summary_stages:
            main_logger.info(f"[INFO] Summaries limited to stages: {enable_latent_summary_stages}")
        else:
            main_logger.info("[INFO] Summaries enabled for ALL stages")
        if enable_summary_cache:
            main_logger.info("[INFO] Summary caching ENABLED - similar prompts will reuse summaries")
        else:
            main_logger.info("[INFO] Summary caching DISABLED")
            main_logger.info("[WARNING] Without caching, API costs will be higher")

    pipeline_success = False
    total_start_time = time.time()

    try:
        # ==================== STAGE 1: Problem Analysis ====================
        tracker.start_stage("Problem Analysis", stage_num=1)
        logger_manager.log_stage_start("Problem Analysis", stage_num=1)
        start_stage1 = time.time()

        try:
            problem, order, with_code, coordinator, task_descriptions, solution = \
                problem_analysis(llm, problem_path, config, dataset_path, output_dir, logger_manager)

            duration_stage1 = int(time.time() - start_stage1)
            tracker.end_stage("Problem Analysis", duration_stage1)
            logger_manager.log_stage_complete("Problem Analysis", duration_stage1, stage_num=1)

            # STEP 4: Record stage completion for failure handling
            if failure_handler:
                failure_handler.record_stage_completion("Problem Analysis")
        except Exception as e:
            tracker.track_error("stage_error", str(e), "Problem Analysis")
            logger_manager.log_exception(e, "Problem Analysis failed", stage="Problem Analysis")
            raise

        # ==================== STAGE 2 & 3: Mathematical Modeling & Computational Solving ====================
        tracker.start_stage("Mathematical Modeling & Computational Solving", stage_num=2)
        logger_manager.log_stage_start("Mathematical Modeling & Computational Solving", stage_num=2)
        start_stage23 = time.time()

        try:
            for id in order:
                task_start_time = time.time()

                # Track task start
                tracker.start_task(id)
                logger_manager.log_progress(f"Task {id}: Starting Mathematical Modeling", level='info')

                # Mathematical Modeling
                task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt = \
                    mathematical_modeling(id, problem, task_descriptions, llm, config, coordinator, with_code, logger_manager)

                # Computational Solving
                solution = computational_solving(
                    llm, coordinator, with_code, problem, id, task_description,
                    task_analysis, task_modeling_formulas, task_modeling_method,
                    dependent_file_prompt, config, solution, name, output_dir, dataset_dir, logger_manager
                )

                task_duration = int(time.time() - task_start_time)
                tracker.end_task(id, result_summary={"duration": task_duration})
                logger_manager.log_progress(f"Task {id} Complete ({task_duration}s)", level='info')

            duration_stage23 = int(time.time() - start_stage23)
            tracker.end_stage("Mathematical Modeling & Computational Solving", duration_stage23)
            logger_manager.log_stage_complete("Mathematical Modeling & Computational Solving", duration_stage23, stage_num=2)

            # STEP 4: Record stage completion for failure handling
            if failure_handler:
                failure_handler.record_stage_completion("Mathematical Modeling & Computational Solving")
        except Exception as e:
            tracker.track_error("stage_error", str(e), "Mathematical Modeling & Computational Solving")
            logger_manager.log_exception(e, "Mathematical Modeling & Computational Solving failed",
                                        stage="Mathematical Modeling & Computational Solving")
            raise

        # ==================== STAGE 4: Solution Reporting (with PDF generation) ====================
        tracker.start_stage("Solution Reporting", stage_num=4)
        logger_manager.log_stage_start("Solution Reporting", stage_num=4)
        start_stage4 = time.time()

        try:
            main_logger.info("[INFO] Stage 4: Generating solution paper (JSON, Markdown, LaTeX, PDF)...")
            paper = generate_paper(llm, output_dir, name)

            duration_stage4 = int(time.time() - start_stage4)
            tracker.end_stage("Solution Reporting", duration_stage4)
            logger_manager.log_stage_complete("Solution Reporting", duration_stage4, stage_num=4)

            # STEP 4: Record stage completion for failure handling
            if failure_handler:
                failure_handler.record_stage_completion("Solution Reporting")

            main_logger.info(f"[OK] Stage 4 completed in {duration_stage4}s")
            main_logger.info("[INFO] Output formats: JSON, Markdown, LaTeX, PDF (if pdflatex available)")

        except Exception as e:
            tracker.track_error("stage_error", str(e), "Solution Reporting")
            logger_manager.log_exception(e, "Solution Reporting failed", stage="Solution Reporting")
            main_logger.warning(f"[WARNING] Stage 4 failed: {e}")
            main_logger.warning("[WARNING] Stages 1-3 completed successfully, but solution reporting failed")
            # Don't raise - graceful degradation, earlier stages succeeded

        # ==================== FINALIZE ====================
        pipeline_success = True
        total_duration = int(time.time() - total_start_time)

        # Log final results
        main_logger.info(f"\n{'='*80}")
        main_logger.info(f"PIPELINE COMPLETE - Total duration: {total_duration}s")
        main_logger.info(f"{'='*80}\n")

        # Print solution summary
        main_logger.info(f"Solution keys: {list(solution.keys())}")
        main_logger.info(f"Usage: {llm.get_total_usage()}")

        # Save usage data
        write_json_file(f'{output_dir}/usage/{name}.json', llm.get_total_usage())

        return solution

    finally:
        # Finalize tracking and run evaluation
        tracker.finalize()

        # CRITICAL FIX: Always run evaluation, regardless of pipeline_success
        # This ensures evaluation folder is generated even if pipeline fails or is interrupted
        try:
            # Use MMBench evaluation with the main model
            main_logger.info(f"\n[INFO] Starting MMBench evaluation with main model: {config['model_name']}...")
            solution_path = Path(output_dir) / "json" / f"{name}.json"
            evaluation_report = run_mmbench_evaluation(key, config['model_name'], str(solution_path), output_dir, logger_manager)
            main_logger.info(f"[OK] MMBench evaluation complete: {evaluation_report}")
        except FileNotFoundError as e:
            # Solution file not found - pipeline likely failed early
            main_logger.warning(f"[WARNING] Solution file not found, skipping MMBench evaluation: {e}")
            # Fall back to internal auto_evaluation which can handle missing files
            try:
                main_logger.info("\n[INFO] Falling back to automatic quality evaluation...")
                evaluation_report = run_auto_evaluation(llm, output_dir, name)
                main_logger.info(f"[OK] Evaluation complete: {evaluation_report}")
            except Exception as e2:
                main_logger.warning(f"[WARNING] Auto evaluation also failed: {e2}")
        except Exception as e:
            main_logger.warning(f"[WARNING] Evaluation failed: {e}")

        # Shutdown logging system
        logger_manager.shutdown()


def check_existing_runs(task_id: str, max_runs: int = 3, method_name: str = 'MM-Agent'):
    """
    Check if there are too many existing runs for the same task.

    P1 FIX: Prevents wasted API calls and storage from duplicate runs.

    Args:
        task_id: The task/problem ID (e.g., "2024_C")
        max_runs: Maximum number of runs before showing warning (default: 3)
        method_name: Method name for output directory (default: "MM-Agent")

    Returns:
        bool: True if should continue, False if user chose to abort
    """
    from pathlib import Path

    output_dir = Path('MMAgent/output') / method_name
    if not output_dir.exists():
        return True  # No output directory yet, first run

    # Count existing runs for this task
    existing_runs = [
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith(task_id)
    ]

    if len(existing_runs) >= max_runs:
        print(f"\n[WARNING] Detected {len(existing_runs)} previous runs for task '{task_id}'")
        print(f"[INFO] Most recent runs:")

        # Sort by modification time (newest first)
        runs_sorted = sorted(existing_runs,
                           key=lambda p: p.stat().st_mtime,
                           reverse=True)

        for i, run in enumerate(runs_sorted[:5], 1):
            # Extract timestamp from directory name
            # Format: taskid_YYYYMMDD-HHMMSS
            parts = run.name.split('_')
            if len(parts) >= 2:
                timestamp = parts[-1]
                # Format timestamp: YYYYMMDD-HHMMSS -> YYYY-MM-DD HH:MM:SS
                if len(timestamp) == 15 and timestamp[8] == '-':
                    formatted_ts = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
                else:
                    formatted_ts = timestamp
                print(f"  {i}. {run.name} ({formatted_ts})")
            else:
                print(f"  {i}. {run.name}")

        print(f"\n[RISK] Running again will:")
        print(f"  - Consume additional API quota (potentially expensive)")
        print(f"  - Create more output files (disk space)")
        print(f"  - May produce the same results if nothing changed")

        # Check if running in non-interactive environment
        response = input("\nContinue anyway? (y/N): ").strip()
        if response.lower() != 'y':
            print("[ABORTED] Run cancelled by user")
            return False

    return True


def preflight_checks(task_id: str):
    """
    Perform pre-flight checks to ensure all required files exist.

    P1 FIX: Validates input files before starting pipeline to prevent
    mid-execution failures that waste API calls.

    Args:
        task_id: The task/problem ID (e.g., "2024_C")

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If problem JSON is malformed
    """
    from pathlib import Path

    print("\n" + "=" * 80)
    print("PREFLIGHT CHECKS")
    print("=" * 80)

    # Check 1: Problem file exists
    problem_path = Path(f'MMBench/problem/{task_id}.json')
    print(f"\n[CHECK 1] Problem file: {problem_path}")

    if not problem_path.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_path}")

    print(f"[OK] Problem file exists")

    # Check 2: Load and validate problem JSON
    print(f"\n[CHECK 2] Validating problem JSON structure...")
    try:
        with open(problem_path, 'r', encoding='utf-8') as f:
            problem_data = __import__('json').load(f)

        # Check required fields
        required_fields = ['background', 'problem_requirement']
        missing_fields = [field for field in required_fields if field not in problem_data]

        if missing_fields:
            raise ValueError(f"Problem JSON missing required fields: {missing_fields}")

        print(f"[OK] Problem JSON is valid")

        # Check 3: Dataset files (if specified)
        if 'dataset_path' in problem_data:
            dataset_paths = problem_data['dataset_path']
            if isinstance(dataset_paths, str):
                dataset_paths = [dataset_paths]

            print(f"\n[CHECK 3] Validating dataset files...")
            for ds_path in dataset_paths:
                # Dataset path can be absolute or relative to MMBench/dataset/
                if Path(ds_path).is_absolute():
                    full_path = Path(ds_path)
                else:
                    full_path = Path('MMBench/dataset') / task_id / ds_path

                print(f"  Checking: {full_path}")

                if not full_path.exists():
                    raise FileNotFoundError(f"Dataset file not found: {ds_path}")

                # Try to read first line to verify file is not corrupted
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                    print(f"  [OK] File is readable")
                except UnicodeDecodeError:
                    # Try with latin1 encoding
                    with open(full_path, 'r', encoding='latin1') as f:
                        first_line = f.readline()
                    print(f"  [OK] File is readable (latin1 encoding)")

            print(f"[OK] All dataset files validated")

        else:
            print(f"\n[CHECK 3] No dataset files specified (skipping)")

        # Check 4: Config file
        print(f"\n[CHECK 4] Configuration file: config.yaml")
        config_path = Path('config.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: config.yaml")

        print(f"[OK] Config file exists")

    except __import__('json').JSONDecodeError as e:
        raise ValueError(f"Problem JSON is malformed: {e}")

    print("\n" + "=" * 80)
    print("[OK] ALL PREFLIGHT CHECKS PASSED")
    print("=" * 80 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                        help='Model name to use (default: gpt-4o)')
    parser.add_argument('--method_name', type=str, default='MM-Agent',
                        help='Method name for output directory (default: MM-Agent)')
    parser.add_argument('--task', type=str, default='2024_C',
                        help='Task/problem ID (default: 2024_C)')
    parser.add_argument('--key', type=str, default='',
                        help='API key for LLM')
    parser.add_argument('--enable_latent_summary', action='store_true',
                        help='Enable LLM-based summaries for each API call (increases cost)')
    parser.add_argument('--latent_summary_stages', type=str, nargs='*', default=None,
                        help='Only generate summaries for specific stages (e.g., "Problem Analysis" "Mathematical Modeling"). '
                             'If not specified, summaries are generated for all stages.')
    parser.add_argument('--disable_summary_cache', action='store_true',
                        help='Disable summary caching (default: enabled). Caching reduces cost by reusing summaries for similar prompts.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # P1 FIX: Run pre-flight checks before starting pipeline
    try:
        preflight_checks(args.task)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] Preflight checks failed: {e}")
        print("[INFO] Please fix the above issues before running the pipeline.")
        import sys
        sys.exit(1)

    # P1 FIX: Check for duplicate runs
    if not check_existing_runs(args.task, max_runs=3, method_name=args.method_name):
        import sys
        sys.exit(0)  # User chose to abort

    # Get experiment info and initialize logging system
    problem_path, config, dataset_dir, output_dir, logger_manager = get_info(args)

    # STEP 0 FIX: Initialize DataManager to lock CSV paths
    # This prevents LLM from generating code with full Windows paths like C:\Users\...
    from utils.data_manager import init_data_manager
    data_manager = init_data_manager(dataset_dir)

    # Pre-flight check: Validate data files exist before starting
    if logger_manager:
        logger_manager.log_progress("Running data file pre-flight check...")

    import json
    with open(problem_path, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)

    required_files = []
    if 'dataset_path' in problem_data:
        dataset_paths = problem_data['dataset_path']
        if isinstance(dataset_paths, str):
            required_files = [Path(dataset_paths).name]
        else:
            required_files = [Path(p).name for p in dataset_paths]

    preflight_result = data_manager.preflight_check(required_files)

    # Report pre-flight results
    print(f"\n{'='*60}")
    print("DATA PRE-FLIGHT CHECK")
    print(f"{'='*60}")
    print(f"DATA_DIR: {preflight_result['data_dir_path']}")
    print(f"CSV files found: {len(preflight_result['csv_files'])}")
    for f in preflight_result['csv_files']:
        print(f"  - {f}")

    if preflight_result['missing_files']:
        print(f"\n[WARNING] Missing required files:")
        for f in preflight_result['missing_files']:
            print(f"  - {f}")

    if preflight_result['unreadable_files']:
        print(f"\n[ERROR] Unreadable files:")
        for f in preflight_result['unreadable_files']:
            print(f"  - {f}")

    print(f"{'='*60}\n")

    # P1 FIX: Run experiment with try-finally to ensure runtime is always saved
    # STEP 4 FIX: Also ensure minimal solution.json is written on failure
    # This prevents cascading crashes when downstream stages look for solution.json
    start = time.time()
    solution = None

    # STEP 4: Initialize FailureHandler for minimal JSON fallback
    from utils.failure_handler import FailureHandler
    failure_handler = FailureHandler(output_dir, args.task)

    try:
        solution = run(
            key=args.key,
            problem_path=problem_path,
            config=config,
            name=args.task,
            dataset_path=dataset_dir,
            output_dir=output_dir,
            logger_manager=logger_manager,
            enable_latent_summary=args.enable_latent_summary,
            enable_latent_summary_stages=args.latent_summary_stages,
            enable_summary_cache=not args.disable_summary_cache,
            failure_handler=failure_handler  # STEP 4: Pass to run() for stage tracking
        )
    except Exception as pipeline_error:
        # STEP 4: Pipeline crashed - write minimal solution.json
        error_msg = f"{type(pipeline_error).__name__}: {str(pipeline_error)}"
        print(f"\n[CRITICAL] Pipeline crashed: {error_msg}")
        print("[STEP 4 FIX] Writing minimal solution.json to prevent cascading failures...")

        import traceback
        traceback_details = traceback.format_exc()

        failure_handler.write_minimal_solution(
            stage='main_pipeline',
            error=error_msg,
            partial_data={
                'exception_type': type(pipeline_error).__name__,
                'traceback': traceback_details
            }
        )

        print(f"[OK] Minimal solution written: {failure_handler.solution_path}")
        print("[INFO] Downstream stages can now read partial results")

        # Re-raise exception after writing minimal solution
        raise

    finally:
        # Always save runtime, even if run() crashes
        end = time.time()
        try:
            # Ensure usage directory exists
            usage_dir = Path(output_dir) / 'usage'
            usage_dir.mkdir(parents=True, exist_ok=True)

            # Write runtime
            with open(usage_dir / 'runtime.txt', 'w', encoding='utf-8') as f:
                f.write("{:.2f}s".format(end - start))

            if solution is None:
                print(f"\n[WARNING] Pipeline crashed, but runtime saved: {end - start:.2f}s")
                print("[INFO] Check logs in output directory for error details")
                print("[INFO] Minimal solution.json has been written to prevent cascading failures")
        except Exception as e:
            print(f"\n[ERROR] Failed to save runtime.txt: {e}")
