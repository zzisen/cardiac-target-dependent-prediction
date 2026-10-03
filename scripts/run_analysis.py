"""Run a frozen analysis, or replay expensive steps from frozen numerical inputs."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/common/release_analysis.py'),run_name='__main__')
