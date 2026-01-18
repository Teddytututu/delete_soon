import os
from pathlib import Path
import pandas as pd
from utils.utils import save_solution
from agent.task_solving import TaskSolver
from agent.create_charts import ChartCreator

# P1 FIX #4: Import BOM cleaning and column normalization
from utils.data_normalization import normalize_dataframe_columns, read_csv_with_normalized_columns

# P0 FIX #1: Import schema normalization utilities for crash prevention
from utils.schema_normalization import normalize_chart

# CRITICAL FIX #5: canonical_whitelist.py is DEPRECATED
# Now using unified DataManager.CANONICAL_INPUT_FILES (single source of truth)
# from utils.canonical_whitelist import CSVWhitelistManager, CANONICAL_CSV_FILES

# 问题3修复: Import shared path AutoFix module
from utils.path_autofix import apply_path_autofix_with_validation

# MINIMAL CHART FIX: Use ChartCreator for chart generation
# Note: MinimalChartCreator was removed, using standard ChartCreator instead
from agent.create_charts import ChartCreator as MinimalChartCreator


def computational_solving(llm, coordinator, with_code, problem, task_id, task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt, config, solution, name, output_dir, dataset_dir, logger_manager=None):
    """Perform computational solving and chart generation with optional logging.

    Args:
        llm: Language model instance
        coordinator: Coordinator instance
        with_code: Whether code execution is enabled
        problem: Problem dictionary
        task_id: Task identifier
        task_description: Task description
        task_analysis: Task analysis
        task_modeling_formulas: Modeling formulas
        task_modeling_method: Modeling method
        dependent_file_prompt: Dependent file prompt
        config: Configuration dictionary
        solution: Solution dictionary
        name: Solution name
        output_dir: Output directory
        logger_manager: Optional MMExperimentLogger instance

    Returns:
        Updated solution dictionary
    """
    # Get logger if available
    if logger_manager:
        main_logger = logger_manager.get_logger('main')
        logger_manager.log_progress(f"Task {task_id}: Code Generation...")

    # CRITICAL FIX: Pass logger_manager to agents for proper error logging to errors.log
    # LATENT REPORTER INTEGRATION: Pass output_dir and task_id for LLM-powered narrative logging
    ts = TaskSolver(llm, logger_manager, output_dir=output_dir, task_id=str(task_id))
    cc = ChartCreator(llm, logger_manager, output_dir=output_dir, task_id=str(task_id))

    # CRITICAL FIX: Use module-relative path to avoid FileNotFoundError when running from different directories
    # Same fix as retrieve_method.py - use Path(__file__).parent to resolve paths correctly
    module_dir = Path(__file__).parent.parent
    code_template_path = module_dir / 'code_template' / f'main{task_id}.py'

    if not code_template_path.exists():
        raise FileNotFoundError(f"Code template not found: {code_template_path}")

    # [NEW] Three-tier structure: code goes to Workspace/code
    work_dir = os.path.join(output_dir, 'Workspace', 'code')
    script_name = 'main{}.py'.format(task_id)
    save_path = os.path.join(work_dir, script_name)

    # P1-5 ENHANCEMENT: Extract column names with Schema Registry
    # This provides centralized schema tracking and validation
    data_columns_info = ""
    schema_registry_path = None  # Will be set if we want to persist

    if logger_manager:
        logger_manager.log_progress(f"Task {task_id}: Scanning data files with Schema Registry...")
        main_logger = logger_manager.get_logger('main')

    try:
        # P1-5: Use Schema Registry for centralized schema management
        from utils.schema_registry import SchemaRegistry
        from utils.data_manager import get_data_manager
        from utils.schema_manager import get_schema_manager

        # Initialize schema registry
        schema_registry = SchemaRegistry(registry_path=schema_registry_path)
        data_mgr = get_data_manager()
        schema_mgr = get_schema_manager()

        scanned_files_info = []

        # CRITICAL FIX #5: Removed hardcoded canonical_files
        # Now using unified DataManager.CANONICAL_INPUT_FILES (single source of truth)

        # CRITICAL FIX #5: Use DataManager's unified whitelist
        if data_mgr and data_mgr.data_dir.exists():
            # Use DataManager's CANONICAL_INPUT_FILES (single source of truth)
            canonical_files = data_mgr.CANONICAL_INPUT_FILES
            all_csv_files = data_mgr.list_csv_files()

            # Filter to only canonical files
            csv_files = [f for f in all_csv_files if f in canonical_files]

            if logger_manager:
                main_logger.info(f"[CRITICAL FIX #5] Filtered to {len(csv_files)} canonical files (from {len(all_csv_files)} total)")
                main_logger.debug(f"[CRITICAL FIX #5] Canonical files: {sorted(canonical_files)}")
                excluded = [f for f in all_csv_files if f not in canonical_files]
                if excluded:
                    main_logger.debug(f"[CRITICAL FIX #5] Excluded non-canonical files: {sorted(excluded)}")

            # P1-5: Scan only canonical CSV files and build schema registry
            input_count = 0
            for filename in csv_files:
                full_path = data_mgr.get_full_path(filename)
                if full_path.exists():
                    count = schema_registry.scan_files(
                        [full_path],
                        source='input_data',
                        max_rows=100
                    )
                    input_count += count

            if input_count > 0 and logger_manager:
                main_logger.info(f"[P1-5] Schema Registry: Scanned {input_count} input files")
                stats = schema_registry.get_statistics()
                main_logger.info(f"[P1-5] Registry Stats: {stats['total_files']} files, {stats['total_columns']} total columns")

            if csv_files:
                if logger_manager:
                    logger_manager.log_progress(f"Found {len(csv_files)} CSV files in DATA_DIR")

                for filename in csv_files:
                    try:
                        # Use DataManager to read file
                        full_path = data_mgr.get_full_path(filename)

                        # P1 FIX #4 + P0-ENCODING FIX: Use read_csv_with_normalized_columns
                        # Now uses utf-8-sig by default (handles BOM), fallback to latin1
                        df_sample = read_csv_with_normalized_columns(
                            full_path,
                            nrows=5
                        )

                        # STEP 1: Normalize column names with SchemaManager (double normalization for safety)
                        normalized_columns = []
                        for col in df_sample.columns:
                            normalized = schema_mgr.normalize_column_name(col)
                            normalized_columns.append(normalized)

                        # Store file info
                        file_info = {
                            'name': filename,
                            'path': str(full_path),
                            'source': 'input_data',
                            'columns': df_sample.columns.tolist(),  # Original columns
                            'normalized_columns': normalized_columns,  # Normalized for LLM
                            'shape_preview': f"{df_sample.shape[0]+1}+ rows × {df_sample.shape[1]} columns"
                        }

                        # Build column details with sample values
                        column_details = {}
                        for i, col in enumerate(df_sample.columns):
                            dtype = str(df_sample[col].dtype)
                            sample_vals = df_sample[col].dropna().head(3).tolist()
                            sample_vals_str = ', '.join(str(v) for v in sample_vals)
                            column_details[col] = f"    - '{col}' ({dtype}): [{sample_vals_str}]\n"

                        file_info['column_details'] = column_details
                        scanned_files_info.append(file_info)

                        # Log for debugging
                        if logger_manager:
                            main_logger = logger_manager.get_logger('main')
                            main_logger.debug(f"Sampled INPUT data from {filename}: {df_sample.columns.tolist()}")

                    except Exception as e:
                        if logger_manager:
                            main_logger = logger_manager.get_logger('main')
                            main_logger.warning(f"Failed to read input CSV {csv_file}: {e}")

        # Part 2: Scan work_dir for task-generated CSV files (CRITICAL for chart generation)
        # P1-5: Use Schema Registry for task-generated files
        if os.path.exists(work_dir):
            task_csv_files = []
            for f in os.listdir(work_dir):
                if f.endswith('.csv'):
                    task_csv_files.append(os.path.join(work_dir, f))

            # P1-5: Scan task-generated files and add to registry
            if task_csv_files:
                task_count = schema_registry.scan_files(
                    task_csv_files,
                    source='task_output',
                    max_rows=100
                )

                if task_count > 0 and logger_manager:
                    main_logger.info(f"[P1-5] Schema Registry: Scanned {task_count} task-generated files")
                    stats = schema_registry.get_statistics()
                    main_logger.info(f"[P1-5] Registry Stats: {stats['total_files']} files, {stats['total_columns']} total columns")

                # Also extract info for compatibility with existing code
                for csv_file in task_csv_files:
                    try:
                        # P1 FIX #4 + P0-ENCODING FIX: Use read_csv_with_normalized_columns
                        # Now uses utf-8-sig by default (handles BOM), fallback to latin1
                        df_sample = read_csv_with_normalized_columns(
                            csv_file,
                            nrows=5
                        )
                        filename = os.path.basename(csv_file)

                        # STEP 1: Normalize column names (already normalized, but apply schema_mgr for consistency)
                        normalized_columns = []
                        for col in df_sample.columns:
                            normalized = schema_mgr.normalize_column_name(col)
                            normalized_columns.append(normalized)

                        # Store file info
                        file_info = {
                            'name': filename,
                            'path': csv_file,
                            'source': 'task_output',
                            'columns': df_sample.columns.tolist(),  # Original
                            'normalized_columns': normalized_columns,  # Normalized
                            'shape_preview': f"{df_sample.shape[0]+1}+ rows × {df_sample.shape[1]} columns"
                        }

                        # Build column details with sample values
                        column_details = {}
                        for col in df_sample.columns:
                            dtype = str(df_sample[col].dtype)
                            sample_vals = df_sample[col].dropna().head(3).tolist()
                            sample_vals_str = ', '.join(str(v) for v in sample_vals)
                            column_details[col] = f"    - '{col}' ({dtype}): [{sample_vals_str}]\n"

                        file_info['column_details'] = column_details
                        scanned_files_info.append(file_info)

                        # Log for debugging
                        if logger_manager:
                            main_logger = logger_manager.get_logger('main')
                            main_logger.debug(f"Sampled TASK OUTPUT from {filename}: {df_sample.columns.tolist()}")

                    except Exception as e:
                        if logger_manager:
                            main_logger = logger_manager.get_logger('main')
                            main_logger.warning(f"Failed to read task output CSV {csv_file}: {e}")

        # Build the final data_columns_info string with both sources
        # P1-5: Use Schema Registry to generate comprehensive schema information
        if schema_registry.schemas:
            # Use Schema Registry's formatted output (more comprehensive)
            data_columns_info = schema_registry.format_for_prompt(max_files=20, max_columns=30)

            if logger_manager:
                stats = schema_registry.get_statistics()
                main_logger.info(f"[P1-5] Generated schema info for {stats['total_files']} files ({stats['input_files']} input + {stats['task_files']} task-generated)")
                main_logger.info(f"[P1-5] Total columns available: {stats['total_columns']}")
        elif scanned_files_info:
            # Fallback to original format if registry is empty
            data_columns_info = "\n## Available Data Files and Columns:\n"

            # Group by source for clarity
            input_files = [f for f in scanned_files_info if f['source'] == 'input_data']
            task_files = [f for f in scanned_files_info if f['source'] == 'task_output']

            if input_files:
                data_columns_info += "\n### Input Data Files (from problem dataset):\n"
                for file_info in input_files:
                    data_columns_info += f"- {file_info['name']}:\n"
                    # STEP 1: Show both original and normalized columns
                    data_columns_info += f"  Original Columns: {file_info['columns']}\n"
                    if 'normalized_columns' in file_info:
                        data_columns_info += f"  Use These Names: {file_info['normalized_columns']}\n"
                    data_columns_info += f"  Shape: {file_info['shape_preview']}\n"
                    data_columns_info += f"  Column Details:\n"
                    for col_detail in file_info['column_details'].values():
                        data_columns_info += col_detail

            if task_files:
                data_columns_info += "\n### Task-Generated Files (from pipeline execution):\n"
                for file_info in task_files:
                    data_columns_info += f"- {file_info['name']}:\n"
                    # STEP 1: Show both original and normalized columns
                    data_columns_info += f"  Original Columns: {file_info['columns']}\n"
                    if 'normalized_columns' in file_info:
                        data_columns_info += f"  Use These Names: {file_info['normalized_columns']}\n"
                    data_columns_info += f"  Shape: {file_info['shape_preview']}\n"
                    data_columns_info += f"  Column Details:\n"
                    for col_detail in file_info['column_details'].values():
                        data_columns_info += col_detail

            data_columns_info += "\n## CRITICAL INSTRUCTIONS:\n"
            data_columns_info += "1. Use ONLY the column names listed above under 'Use These Names'\n"
            data_columns_info += "2. Do NOT guess or invent column names - they will cause KeyError!\n"
            data_columns_info += "3. The sample values show REAL data - use them to understand the format\n"
            data_columns_info += "4. Column names are case-sensitive: 'Event' != 'event' != 'EVENT'\n"

            if logger_manager:
                main_logger.info(f"Extracted column and sample information from {len(scanned_files_info)} files ({len(input_files)} input + {len(task_files)} task-generated)")
    except Exception as e:
        if logger_manager:
            main_logger = logger_manager.get_logger('main')
            main_logger.warning(f"Failed to extract column/sample information: {e}")
        data_columns_info = ""

    if with_code:
        # CRITICAL FIX: Ensure data_path_for_coding is a string path
        # dataset_dir is the actual directory path like 'MMBench/dataset/2000_C'
        # coding() expects a string path for data_file parameter
        data_path_for_coding = str(dataset_dir) if dataset_dir else "."

        # CRITICAL FIX [2026-01-18]: Ensure data_description is a string
        # problem['data_description'] in JSON is often a dict, which can cause
        # issues when passed to safe_format(). Convert to JSON string if needed.
        data_desc = problem.get('data_description', '')
        if isinstance(data_desc, (dict, list)):
            import json
            data_desc_str = json.dumps(data_desc, indent=2, ensure_ascii=False)
        else:
            data_desc_str = str(data_desc)

        # CRITICAL FIX [2026-01-18]: Ensure variable_description is also a string
        var_desc = problem.get('variable_description', '')
        if isinstance(var_desc, (dict, list)):
            import json
            var_desc_str = json.dumps(var_desc, indent=2, ensure_ascii=False)
        else:
            var_desc_str = str(var_desc)

        task_code, is_pass, execution_result = ts.coding(
            data_path_for_coding,  # FIXED: Pass directory path string, not list
            data_desc_str,  # FIXED: Ensure string format (JSON if dict/list)
            var_desc_str,  # FIXED: Ensure string format (JSON if dict/list)
            task_description,
            task_analysis,
            task_modeling_formulas,
            task_modeling_method,
            dependent_file_prompt,
            code_template,
            script_name,
            work_dir,
            user_prompt=data_columns_info  # CRITICAL: Pass actual column names to LLM
        )
        code_structure = ts.extract_code_structure(task_id, task_code, save_path)
        task_result = ts.result(task_description, task_analysis, task_modeling_formulas, task_modeling_method, execution_result)
        task_answer = ts.answer(task_description, task_analysis, task_modeling_formulas, task_modeling_method, task_result)

        # Log code execution result
        if logger_manager:
            logger_manager.log_code_execution(
                task_id=task_id,
                script_name=script_name,
                stdout=str(execution_result),
                stderr="",
                return_code=0 if is_pass else 1
            )

        task_dict = {
            'task_description': task_description,
            'task_analysis': task_analysis,
            'modeling_formulas': task_modeling_formulas,  # Changed from preliminary_formulas
            'mathematical_modeling_process': task_modeling_method,
            'task_code': task_code,
            'is_pass': is_pass,
            'execution_result': execution_result,
            'solution_interpretation': task_result,
            'subtask_outcome_analysis': task_answer
        }
        coordinator.code_memory[str(task_id)] = code_structure
    else:
        task_result = ts.result(task_description, task_analysis, task_modeling_formulas, task_modeling_method)
        task_answer = ts.answer(task_description, task_analysis, task_modeling_formulas, task_modeling_method, task_result)
        task_dict = {
            'task_description': task_description,
            'task_analysis': task_analysis,
            'modeling_formulas': task_modeling_formulas,  # Changed from preliminary_formulas
            'mathematical_modeling_process': task_modeling_method,
            'solution_interpretation': task_result,
            'subtask_outcome_analysis': task_answer
        }
    coordinator.memory[str(task_id)] = task_dict

    # Generate charts with actual images
    if logger_manager:
        logger_manager.log_progress(f"Task {task_id}: Chart Generation... ({config['num_charts']} charts)")

    # CRITICAL FIX #5: Use unified DataManager for file filtering
    # Get canonical input files (whitelist managed)
    input_paths = [str(f) for f in data_mgr.get_canonical_input_files()]
    # Extract just filenames for comparison when filtering generated files
    input_data_files = [os.path.basename(p) for p in input_paths]

    # Get task-generated output files (auto-discovered)
    output_paths = [str(f) for f in data_mgr.get_task_generated_files()]

    # Combine: output files first (priority for chart generation), then input files
    all_plottable_files = output_paths + input_paths

    if logger_manager:
        main_logger.info(f"[CRITICAL FIX #5] Chart generation file filtering:")
        main_logger.info(f"[CRITICAL FIX #5]   - {len(input_paths)} canonical input files")
        main_logger.info(f"[CRITICAL FIX #5]   - {len(output_paths)} task-generated files")
        main_logger.info(f"[CRITICAL FIX #5]   - {len(all_plottable_files)} total plottable files")


    # Collect CSV data files generated by the task (EXCLUDE input data files)
    data_files = []
    if os.path.exists(work_dir):
        for f in os.listdir(work_dir):
            # Only include CSV files that are NOT input data
            # This ensures charts use results generated by the pipeline, not raw input
            if f.endswith('.csv') and f not in input_data_files:
                data_files.append(os.path.join(work_dir, f))

    if logger_manager:
        main_logger.debug(f"[P0-1] Canonical input files: {input_data_files}")
        main_logger.debug(f"[P0-1] Generated result files: {len(data_files)} files")

    # MINIMAL CHART FIX: Use minimal chart creator instead of complex cc.create_charts_with_images
    # This implements the "减法修复" from chat with claude2.txt and chat with claude3.txt
    # Key changes:
    # 1. No guards, no AST, no autofix - just matplotlib
    # 2. Success = file.exists() and file.stat().st_size > 0 (objective truth)
    # 3. Minimal JSON: {title, image_path, success} (3 fields vs 7+)
    # 4. Single save point at framework layer only

    # LATENT REPORTER INTEGRATION: Pass output_dir and task_id for visual insights logging
    minimal_cc = MinimalChartCreator(llm, logger_manager, output_dir=output_dir, task_id=str(task_id))

    # Generate simple chart descriptions (one per chart)
    # In production, these could be generated by LLM, but for now use simple templates
    chart_descriptions = [
        f"Chart {i} for task {task_id} showing data visualization"
        for i in range(1, config['num_charts'] + 1)
    ]

    # [NEW] Three-tier structure: charts go to Workspace/charts
    charts_dir = os.path.join(output_dir, 'Workspace', 'charts')
    charts = minimal_cc.create_charts_from_csv_files(
        chart_descriptions=chart_descriptions,
        save_dir=charts_dir,
        csv_files=data_files if data_files else all_plottable_files,  # CRITICAL FIX #5: Use unified file list
        data_columns_info=data_columns_info
    )

    # MINIMAL CHART FIX: Normalize charts to minimal structure
    # The minimal chart creator already returns {title, image_path, success}
    # But we apply normalize_chart() for safety and backward compatibility
    charts = [normalize_chart(c) for c in charts]

    # Log chart generation results
    if logger_manager:
        for i, chart in enumerate(charts):
            # Minimal JSON uses 'title' instead of 'description'
            description = chart.get('title', chart.get('description', f'Chart {i+1}'))
            logger_manager.log_chart_generation(
                task_id=task_id,
                num_charts=i+1,
                description=description,
                success=chart.get('success', False),
                error=chart.get('error')
            )

    # [NEW] Issue 18 FIX: Enhanced chart quality checking with WARNING status
    # Detect suboptimal charts (too small, potentially empty, or quality issues)
    processed_charts = []
    for i, chart in enumerate(charts):
        chart_info = {
            "title": chart.get('title', f'Chart {i+1}'),
            "path": chart.get('image_path', ''),
            "success": chart.get('success', False)
        }

        # Quality check: If file is too small, mark as WARNING
        if chart.get('success') and chart.get('image_path'):
            image_path = chart['image_path']
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)

                # Less than 1KB is likely an empty or corrupted chart
                if file_size < 1024:
                    chart_info["status"] = "WARNING"
                    chart_info["issue"] = f"Chart file too small ({file_size} bytes), possibly empty or corrupted"

                    # Log warning for visibility
                    if logger_manager:
                        main_logger = logger_manager.get_logger('main')
                        main_logger.warning(
                            f"[Issue 18] Task {task_id} Chart {i+1}: {chart_info['issue']}"
                        )
                else:
                    chart_info["status"] = "SUCCESS"
                    chart_info["file_size"] = file_size
            else:
                chart_info["status"] = "FAILED"
                chart_info["issue"] = "File not found after successful generation"
        else:
            # Chart generation failed
            chart_info["status"] = "FAILED"
            if chart.get('error'):
                chart_info["issue"] = chart['error']

        processed_charts.append(chart_info)

    # Store original charts in task dict (for backward compatibility)
    task_dict['charts'] = charts
    # [NEW] Store enhanced chart details with quality status
    task_dict['charts_detail'] = processed_charts
    solution['tasks'].append(task_dict)
    save_solution(solution, name, output_dir)

    # Print summary to console (MINIMAL VERSION: binary success/failure)
    # Simplified from three-state (SUCCESS/PARTIAL/FAILED) to binary (success/failed)
    success_count = sum(1 for c in charts if c.get('success', False))
    failed_count = len(charts) - success_count

    if not logger_manager:
        print(f"\n{'='*60}")
        print(f"Generating charts for Task {task_id}")
        print(f"{'='*60}")
        print(f"Found {len(data_files)} data files for chart generation:")
        for f in data_files:
            print(f"  - {os.path.basename(f)}")

        # MINIMAL VERSION: Simple binary statistics
        print(f"\nCharts generated: {success_count}/{len(charts)} successful")
        for i, chart in enumerate(charts):
            image_path = chart.get('image_path', '')
            success_flag = chart.get('success', False)

            # Verify file actually exists (objective truth check)
            file_exists = os.path.exists(image_path) if image_path else False
            file_size = os.path.getsize(image_path) if file_exists else 0

            if success_flag and file_exists and file_size > 0:
                status = "OK"
                path = f"{image_path} ({file_size:,} bytes)"
            else:
                status = "FAIL"
                if success_flag and not file_exists:
                    path = f"Missing file: {image_path}"
                elif file_exists and file_size == 0:
                    path = f"Empty file: {image_path}"
                else:
                    path = chart.get('error', 'Unknown error')
            print(f"   {status} Chart {i+1}: {path}")
    else:
        # MINIMAL VERSION: Log simple binary statistics
        logger_manager.log_progress(
            f"Task {task_id}: Charts generated - "
            f"{success_count}/{len(charts)} successful"
        )

    return solution
