from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python/src'))
sys.path.insert(0, str(ROOT/'examples/agent'))
os.environ.setdefault('ARIADNE_PLUGIN_ROOT', str(ROOT))
