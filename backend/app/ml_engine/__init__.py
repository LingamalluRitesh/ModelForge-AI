"""
ModelForge AI - app.ml_engine Bridge
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_path = str(Path(__file__).resolve().parent.parent.parent)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import ml_engine.algorithms as algorithms
import ml_engine.preprocessing as preprocessing
import ml_engine.evaluation as evaluation
import ml_engine.optimization as optimization
import ml_engine.explainability as explainability
import ml_engine.fairness as fairness
import ml_engine.data_quality as data_quality
import ml_engine.profiling as profiling
import ml_engine.drift as drift
import ml_engine.automl as automl

# Register in sys.modules for submodule access
sys.modules['app.ml_engine.algorithms'] = algorithms
sys.modules['app.ml_engine.preprocessing'] = preprocessing
sys.modules['app.ml_engine.evaluation'] = evaluation
sys.modules['app.ml_engine.optimization'] = optimization
sys.modules['app.ml_engine.explainability'] = explainability
sys.modules['app.ml_engine.fairness'] = fairness
sys.modules['app.ml_engine.data_quality'] = data_quality
sys.modules['app.ml_engine.profiling'] = profiling
sys.modules['app.ml_engine.drift'] = drift
sys.modules['app.ml_engine.automl'] = automl
