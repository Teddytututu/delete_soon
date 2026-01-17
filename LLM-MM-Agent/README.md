# 🤖 MM Agent: LLMs as Agents for Real-world Mathematical Modeling Problems

<p align="center">
   <img src="assets/logo.png" alt="MM-Agent Logo" width="30%">
</p>

> 📖 This is the English version of the README. [点击查看中文版](./README_zh.md)

## 📰 News
1. **2025-12**
   🔥 **Upcoming Update**: We will soon release the latest upgraded version of the demo. Please **Star** 🌟 our repository! We will issue service accounts based on the Star list (due to limited server capacity) to help everyone better prepare for the MCM/ICM contest.
   (即将更新：我们将很快发布最新升级版演示。请 **Star** 🌟 我们的仓库！由于服务器容量有限，我们将根据 Star 列表发放服务账号，帮助大家更好的备战美赛。)
2. **2025-10**
   🚀 **MM-Agent assisted two undergraduate teams in winning the Finalist Award** (Top 2.0% among 27,456 teams) in **MCM/ICM 2025**, demonstrating its practical effectiveness as a *modeling copilot*.
   🔗  [Demo](https://huggingface.co/spaces/MathematicalModelingAgent/MathematicalModelingAgent)
3. **2025-09**
   🎉 Our paper *"MM-Agent: LLMs as Agents for Real-world Mathematical Modeling Problems"* has been accepted to the **NeurIPS 2025**!
   📄 [Read the paper on arXiv](https://arxiv.org/abs/2505.14148)
4. **2025-07**
   🎉 Our paper *"MM-Agent: LLMs as Agents for Real-world Mathematical Modeling Problems"* has been accepted to the **AI4MATH Workshop at ICML 2025**!
   📄 [Read the paper on arXiv](https://arxiv.org/abs/2505.14148)

## 📖 Overview

We propose a **Mathematical Modeling Agent** that simulates the real-world human process of mathematical modeling. This agent follows a complete problem-solving pipeline:

1. **Problem Analysis**
2. **Mathematical Modeling**
3. **Computational Solving**
4. **Solution Reporting**

Our paper is available at [arXiv](https://arxiv.org/abs/2505.14148).

## 🎥 Demo Video

[**▶️ Watch the Demo Video**](assets/demo.mp4)

> 💡 Note: Click the link above to watch the demo on GitHub.

## 🚀 Core Features Walkthrough

### 1. Project Creation
Initialize your modeling workspace effortlessly.
<img src="assets/step1_project_creation.png" width="100%">

### 2. Upload Problem & Data
Simply upload your problem statement and datasets.
<img src="assets/step2_upload_data.png" width="100%">

### 3. Automated Modeling
The agent intelligently selects and builds mathematical models.
<img src="assets/step3_modeling.png" width="100%">

### 4. Data Analysis
Execute complex data analysis and generate visualizations.
<img src="assets/step4_analysis.png" width="100%">

### 5. Paper Writing
Auto-generate professional reports and academic papers.
<img src="assets/step5_paper_writing.png" width="100%">

### 6. Project Management
Track and manage multiple modeling projects efficiently.
<img src="assets/step6_project_management.png" width="100%">

## 🖼️ Framework Overview

<div align="center">
   <img src="figs/Overview.pdf" alt="MM Agent Framework Overview" width="700px"/>
</div>



---

## 🔬 How Does the Mathematical Modeling Agent Work?

The agent simulates a real-world mathematical modeling workflow through the following structured stages:

1. **🧠 Problem Analysis**
   Understands the background, objectives, data availability, and constraints of the problem.

2. **📐 Mathematical Modeling**
   Translates real-world problems into mathematical models using appropriate assumptions, formulations, and modeling techniques.

3. **🧮 Computational Solving**
   Implements algorithms, simulations, or optimization techniques to solve the models, often involving numerical computation.

4. **📝 Solution Reporting**
   Summarizes the full modeling process, interprets results, and generates a clear, structured report.

We propose MM-Agent, an end-to-end solution for open-ended real-world modeling problems. Inspired by expert workflows, MM-Agent systematically analyzes unstructured problem descriptions, formulates structured mathematical models, derives solutions, and generates analytical reports.
Among these stages, the modeling step poses the greatest challenge, as it requires abstracting complex scenarios into mathematically coherent formulations grounded in both problem context and solution feasibility. To address this, we introduce the Hierarchical Mathematical Modeling Library (HMML): a tri-level knowledge hierarchy encompassing domains, subdomains, and method nodes. HMML encodes 98 high-level modeling schemas that enable both problem-aware and solution-aware retrieval of modeling strategies, supporting abstraction and method selection.  Specifically, MM-Agent first analyzes the problem and decomposes it into subtasks. It then retrieves suitable methods from HMML and refines its modeling plans via an actor-critic mechanism. To solve the models, the agent autonomously generates and iteratively improves code using the MLE-Solver for efficient, accurate execution. Finally, it compiles a structured report summarizing the modeling approach, experimental results, and key insights.

---
## 🌐 Demo
Our demo is available at [Hugging Face Spaces](https://huggingface.co/spaces/MathematicalModelingAgent/MathematicalModelingAgent).

---

## 👾 Currently Supported Models

* **OpenAI**: `gpt-4o`, `gpt-4`
* **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`
* **Zhipu AI (GLM)**: `glm-4-plus`, `glm-4-0520`, `glm-4-air`, `glm-4-flash`, `glm-4.7`
* **Qwen**: `qwen2.5-72b-instruct`

---

## ▶️ Quick Start

### 🔧 Running the Agent

**IMPORTANT**: The `--model_name` parameter is **required**.

```bash
python MMAgent/main.py --key "your_api_key" --task "task_id" --model_name "model_name"
```

**Examples**:

```bash
# Using OpenAI GPT-4o
python MMAgent/main.py --key "sk-XXX" --task "2024_C" --model_name "gpt-4o"

# Using GLM-4-flash (faster/cheaper for testing)
python MMAgent/main.py --key "sk-XXX" --task "2024_C" --model_name "glm-4-flash"

# Using DeepSeek
python MMAgent/main.py --key "sk-XXX" --task "2024_C" --model_name "deepseek-chat"
```

Here, `task` corresponds to the problem ID from MM-Bench (e.g., `"2024_C"` refers to the 2024 MCM problem C).

**Working Directory**: Always run from `LLM-MM-Agent/` directory.

---

## 🖥️ Installation Guide

### ✅ Prerequisites

* Python 3.10 recommended
* Conda (optional but preferred)

### 💻 Setup Steps

1. **Clone the Repository**

```bash
git clone git@github.com:usail-hkust/LLM-MM-Agent.git
```

2. **Create and Activate the Conda Environment**

```bash
conda create --name math_modeling python=3.10
conda activate math_modeling
```

3. **Navigate to Project Directory**

```bash
cd MM-Agent
```

4. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Edit `config.yaml` to customize the agent behavior:

```yaml
# Iteration rounds for actor-critic refinement
problem_analysis_round: 3
problem_modeling_round: 3
task_formulas_round: 3

# Number of subtasks to generate
tasknum: 4

# Number of charts per task
chart_num: 3

# Number of methods to retrieve from HMML
top_method_num: 6

# Logging configuration
logging:
  console_level: INFO
  log_to_file: true
  log_rotation: true
  max_log_size_mb: 50
  backup_count: 5
```

**Code Execution Timeout**:
- Default: 300 seconds (5 minutes)
- To modify: Edit `MMAgent/agent/task_solving.py` line 38

---

## 📂 Output Structure

Results are saved to: `MMAgent/output/{model_name}/{task_id}_{timestamp}/`

```
output/
├── json/           # Structured JSON solution
├── markdown/       # Human-readable report with LaTeX equations
├── latex/          # Academic paper format
├── code/           # Generated Python code (main.py, main2.py, etc.)
├── charts/         # Generated visualization images
├── usage/          # API usage statistics
└── logs/           # Comprehensive execution logs
    ├── main.log                    # General execution log
    ├── errors.log                  # Errors and warnings only
    ├── llm_api.log                 # All LLM API calls with token usage
    ├── code_execution.log          # Code execution details
    ├── chart_generation.log        # Chart generation events
    ├── execution_tracker.jsonl     # Line-by-line JSON events
    ├── execution_tracker_complete.json  # All events as JSON
    ├── execution_tracker_readable.txt   # Human-readable report
    └── brief_summary.txt           # Quick overview (START HERE)
```

**Quick Check**: After each run, open `logs/brief_summary.txt` for an overview.

---

## 🔑 API Keys

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional
```

### DeepSeek
```bash
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"  # Optional
```

### GLM (Zhipu AI)
```bash
export ZHIPU_API_KEY="your_id.your_secret"
# API base is hardcoded: https://open.bigmodel.cn/api/paas/v4
```

### Qwen (Alibaba Cloud)
```bash
export DASHSCOPE_API_KEY="sk-..."
# API base is hardcoded: https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## 🔍 Troubleshooting

### Error 429 (API Rate Limiting)

**Symptom**: Crashes with "Error 429: Rate limit exceeded"

**Status**: ✅ FIXED

**Solution**: The system now uses threading locks to serialize API calls. If you still see this error:
- Verify you're using the updated code
- Check that `api_lock` exists in LLM instance
- Verify API key quota

### Code Execution Hangs

**Symptom**: Pipeline hangs indefinitely during code execution

**Status**: ✅ FIXED

**Solution**: Code execution now has 300-second timeout. If it still hangs:
- Check `logs/code_execution.log` for details
- Review generated code in `output/*/code/main*.py`
- Reduce timeout in `task_solving.py:38` if needed

### UnboundLocalError

**Symptom**: `UnboundLocalError: 'new_content' referenced before assignment`

**Status**: ✅ FIXED

**Solution**: Variables are now initialized at function start. Verify:
- `MMAgent/agent/task_solving.py` has proper initialization
- Response validation checks are in place

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'MMAgent.utils.execution_tracker'`

**Solution**: Ensure all files are present:
```bash
ls MMAgent/utils/execution_tracker.py
ls MMAgent/utils/logging_config.py
ls MMAgent/utils/auto_evaluation.py
```

### No Log Files Created

**Symptom**: Output directory exists but `logs/` is missing or empty

**Solution**:
1. Check `config.yaml` has `logging.log_to_file: true`
2. Verify `LoggerManager` is initialized in `main.py`
3. Check permissions on output directory

For more details, see `CLAUDE.md`.

---

## 🧪 Testing

### Quick Verification

```bash
# Test imports (from LLM-MM-Agent directory)
python -c "from MMAgent.llm.llm import LLM; print('✓ Import works')"

# Test with sample problem (requires API key)
python MMAgent/main.py \
  --key "your_api_key" \
  --task "2025_C" \
  --model_name "glm-4-flash"  # Faster/cheaper for testing
```

### Comprehensive Testing

See `test/` directory for full test suite:

```bash
# Run all tests
python test/scripts/run_all_tests.py

# Run quick tests only
python test/scripts/run_all_tests.py --quick
```

---

## 🤝 Contact & Community

Join our WeChat group for updates and service support!
<img src="assets/wechat_group.jpg" width="30%">

## 📜 License

Source code is licensed under the **\[CC BY-NC]**.


## 📚 References

```bash
@misc{mmagent,  
   title={MM-Agent: LLM as Agents for Real-world Mathematical Modeling Problem},  
   author={Fan Liu and Zherui Yang and Cancheng Liu and Tianrui Song and Xiaofeng Gao and Hao Liu},  
   year={2025},  
   eprint={2505.14148},  
   archivePrefix={arXiv},  
   primaryClass={cs.AI},  
   url={https://arxiv.org/abs/2505.14148}  
}
```