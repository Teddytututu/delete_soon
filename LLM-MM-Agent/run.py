"""
MM-Agent Bootstrap Script

[FIX 11.6] This script serves as the standard entry point for MM-Agent.
Environment configuration (sys.path setup) is isolated here, separate from business logic.

Usage:
    python run.py --task your_task_name --model gpt-4o

Alternative (standard Python module launch):
    python -m MMAgent.main --task your_task_name

Author: MM-Agent Development Team
Date: 2026-01-17
"""

import sys
import os


def launch():
    """
    Bootstrap function for MM-Agent.

    This function:
    1. Configures the Python path (sys.path) for module imports
    2. Imports and calls the main entry point from MMAgent.main

    Environment setup is centralized here, keeping MMAgent/main.py clean.
    """
    # 1. Get project root directory (LLM-MM-Agent/)
    # __file__ is the path to this script (run.py)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. Add project root to sys.path
    # This enables: from MMAgent.xxx import yyy (absolute imports)
    # And also supports: from .xxx import yyy (relative imports in MMAgent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 3. Add MMAgent internal directory to sys.path
    # This provides backward compatibility for legacy code that uses absolute imports
    # like "import utils" instead of "from MMAgent.utils import xxx"
    mmagent_dir = os.path.join(project_root, 'MMAgent')
    if mmagent_dir not in sys.path:
        sys.path.insert(0, mmagent_dir)

    # 4. Print diagnostic information
    print(f"[Launcher] Project Root: {project_root}")
    print(f"[Launcher] MMAgent Directory: {mmagent_dir}")
    print(f"[Launcher] Starting MM-Agent...")

    # 5. Import and run the main function
    # Using MMAgent.main.main() instead of direct execution
    # This allows the module to be imported by other code if needed
    try:
        from MMAgent.main import main as mmagent_main
        mmagent_main()
    except ImportError as e:
        print(f"[ERROR] Failed to import MMAgent.main: {e}")
        print("\n[Troubleshooting Tips]")
        print("1. Ensure you are running run.py from the project root directory (LLM-MM-Agent/)")
        print("2. Check that MMAgent/ directory exists in the current location")
        print("3. Try running: python -m MMAgent.main --task your_task")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] MM-Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    launch()
