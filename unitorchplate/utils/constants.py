import os

LIBRARY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(LIBRARY_DIR)
DATAFILES_DIR = os.path.join(PROJECT_DIR, 'datafiles')
LOGGING_DIR = os.path.join(PROJECT_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'selected-checkpoints')
MODEL_DIR = os.path.join(PROJECT_DIR, 'saved-models')
