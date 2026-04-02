# Soft Tree Training and Policy Extraction

This repository is associated with the following paper:

> **Citation**
>
> Moayyedi, S.A., Yang, D.Y., 2026. Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization.

## Overview

The `softtree` package includes classes for both the soft tree classifier (`SoftTreeClassifier` in `softtree_classification.py`) and the oblique decision tree classifier (`ParameterizedObliqueTree` in `oblique_tree.py`).

The current implementation slightly deviates from the descriptions in the cited paper in the following ways:

* **Depth Definition**: The `max_depth` parameter is zero-indexed. Therefore, the actual depth of a tree (as described in the paper) is `max_depth` + 1.

* **Decision Rules:** At each internal node, the decision rule in this implementation dictates that if $\mathbf{w}^\intercal \mathbf{x} + b \geq 0$, the traversal goes to the left branch; otherwise, it goes to the right branch. The cited paper uses the opposite convention. Therefore, the weights and bias in the paper correspond to $-\mathbf{w}$ and $-b$ in this codebase. This inconsistency will be resolved in future versions of the package.

## Replicating Study Results

Use the `classification_st.py` file to replicate the paper's results regarding soft tree models, annealing schedules, and regularization schemes for supervised classification experiments.

Use the `classification_odt.py` file to replicate the results for oblique decision trees, which are derived from the soft tree baselines using pruning routines and different pruning thresholds.
