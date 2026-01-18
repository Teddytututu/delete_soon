import os
import json
import re
import sys
import argparse
from functools import partial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.model import gpt
from evaluation.utils import load_solution_json
from evaluation.prompts import generate_problem_analysis_prompt, generate_modeling_rigorousness_prompt, generate_practicality_and_scientificity_prompt, generate_result_and_bias_analysis_prompt


# parse the evalution result
def parse_and_save_evaluation_data(text):
    reason_pattern = r'<reason>\s*(.*?)\s*</reason>'
    score_pattern = r'<score>\s*(\d+)\s*</score>'

    reasons = re.findall(reason_pattern, text, re.DOTALL)
    scores = re.findall(score_pattern, text, re.DOTALL)

    if not reasons or not scores:
        print("No matches found for reasons or scores!")
        return {}

    if len(reasons) != len(scores):
        print("Mismatch between number of reasons and scores!")
        return {}

    result_dict = {}
    for i, (reason, score) in enumerate(zip(reasons, scores), 1):
        result_dict[f"analysis_{i}"] = {
            "reason": reason.strip(),
            "score": int(score)
        }

    return result_dict


def evaluate_math_modeling(llm, solution_path):
    
    # get file name
    file_name = os.path.basename(solution_path).split('.')[0]

    # define evaluation result directory
    base_dir = os.path.dirname(solution_path)
    evaluation_dir = os.path.join(base_dir, "evaluation_result")
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    if not os.path.exists(os.path.join(evaluation_dir, file_name)):
        os.makedirs(os.path.join(evaluation_dir, file_name))
    evaluation_dir = os.path.join(evaluation_dir, file_name)
    print(f"Evaluation directory created at: {evaluation_dir}")

    # define evaluation result path
    evaluation_results_txt_path = os.path.join(evaluation_dir, 'evaluation_results.txt')
    evaluation_results_json_path = os.path.join(evaluation_dir, 'evaluation_results.json')
    
    # load problem sultion
    solution_data = load_solution_json(solution_path)
    if not solution_data:
        print("Error: can not load the solution file.")
        return

    # =========================================================================
    # [CRITICAL FIX] 数据结构自适应与类型检查
    # =========================================================================
    # Check if solution_data is a list and normalize it
    if isinstance(solution_data, list):
        print("[INFO] Detected list structure for solution, wrapping in 'tasks' dict.")
        solution_data = {"tasks": solution_data}
    elif isinstance(solution_data, dict):
        if "tasks" not in solution_data:
            # Try to handle single task or minimal solutions
            if "problem_analysis" in solution_data or "mathematical_modeling" in solution_data:
                print("[INFO] Single task solution detected, adding 'tasks' wrapper.")
                solution_data = {"tasks": [solution_data]}
            else:
                print("[WARN] Solution dict missing 'tasks' key. Evaluation might fail.")
    else:
        print(f"[ERROR] Invalid solution data type: {type(solution_data)}")
        return
    # =========================================================================

    # evaluation
    problem_analysis_evaluation_prompt = generate_problem_analysis_prompt(solution_data)
    analysis_evaluation_results = llm(problem_analysis_evaluation_prompt)

    modeling_rigorousness_evaluation_prompt = generate_modeling_rigorousness_prompt(solution_data)
    modeling_rigorousness_evaluation_results = llm(modeling_rigorousness_evaluation_prompt)

    practicality_and_scientificity_evaluation_prompt = generate_practicality_and_scientificity_prompt(solution_data)
    practicality_and_scientificity_evaluation_results = llm(practicality_and_scientificity_evaluation_prompt)

    result_and_bias_analysis_evaluation_prompt = generate_result_and_bias_analysis_prompt(solution_data)
    result_and_bias_analysis_evaluation_results = llm(result_and_bias_analysis_evaluation_prompt)

    # parse the evaluation result
    all_evaluation_data = {
        "analysis_evaluation": parse_and_save_evaluation_data(analysis_evaluation_results[0]),
        "modeling_rigorousness_evaluation": parse_and_save_evaluation_data(modeling_rigorousness_evaluation_results[0]),
        "practicality_and_scientificity_evaluation": parse_and_save_evaluation_data(practicality_and_scientificity_evaluation_results[0]),
        "result_and_bias_analysis_evaluation": parse_and_save_evaluation_data(result_and_bias_analysis_evaluation_results[0])
    }

    # save the evaluation result
    with open(evaluation_results_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_evaluation_data, f, ensure_ascii=False, indent=2)

    print(f"The evaluation result is saved in {evaluation_results_json_path}.")

    # transform the result to txt format
    all_evaluation_results = (
        analysis_evaluation_results[0] + "\n\n" +
        modeling_rigorousness_evaluation_results[0] + "\n\n" +
        practicality_and_scientificity_evaluation_results[0] + "\n\n" +
        result_and_bias_analysis_evaluation_results[0]
    )

    # save the evaluation result
    with open(evaluation_results_txt_path, 'w', encoding='utf-8') as f:
        f.write(all_evaluation_results)

    print(f"The evaluation result is saved in {evaluation_results_txt_path}.")


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
        import os
        return os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
    else:
        # Default to OpenAI
        return provided_base_url or "https://api.openai.com/v1"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--solution_file_path',
        type=str,
        default='MMBench/example_solution/example1.json'
    )

    parser.add_argument(
        '--base_url',
        type=str,
        default=None,
        help='API base URL (auto-detected from model if not provided)'
    )

    parser.add_argument(
        '--key',
        type=str,
        default='',
        help='API key for the model'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for evaluation (supports: gpt-4o, gpt-4, glm-4-flash, glm-4-plus, glm-4.7, glm-4-air, qwen2.5-72b-instruct, deepseek-chat, deepseek-reasoner)'
    )

    parser.add_argument(
        '--temp',
        type=float,
        default=0.7,
        help='Temperature setting for the model'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4000,
        help='Maximum number of tokens for the model'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Auto-detect base_url based on model name
    base_url = get_model_base_url(args.model, args.base_url)

    print(f"[INFO] Using model: {args.model}")
    print(f"[INFO] Using base_url: {base_url}")

    llm = partial(gpt, base_url=base_url, key=args.key, model=args.model, temperature=args.temp, max_tokens=args.max_tokens)
    solution_path = args.solution_file_path
    evaluate_math_modeling(llm, solution_path)
