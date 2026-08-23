"""Compatibility accessors for paths centralized in global_config.py."""
from global_config import (ATLAS_DIR, BASE_DIR, FIGURE_DIR, MODEL_DIR,
                           RESULTS_DIR)


base_dir = BASE_DIR


def set_base_dir():
    return BASE_DIR


def set_fusion_dir(base_dir=base_dir):
    return BASE_DIR


def set_atlas_dir(base_dir=base_dir):
    return ATLAS_DIR


def set_model_dir(base_dir=base_dir):
    return MODEL_DIR


def set_export_dir(base_dir=base_dir):
    return ATLAS_DIR


def set_figure_dir():
    return str(FIGURE_DIR)


def set_results_dir(base_dir=base_dir):
    return str(RESULTS_DIR)
