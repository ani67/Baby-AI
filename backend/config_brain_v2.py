"""
Configuration flag for BrainV2 architecture.

Set USE_BRAIN_V2 = True to switch the training pipeline from BrainState
(Forward-Forward only) to BrainV2 (Forward-Forward + REFLECT hybrid).

No changes to train_worker.py are needed — the model class swap happens
in baby_model_v2_reflect.py, which is imported based on this flag.
"""

USE_BRAIN_V2 = True  # BrainV2: distributed representations + REFLECT
