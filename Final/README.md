## Problem
Create a regression tree model such that the leaf node predicts a function instead of a mean. For example, see a new formulation in Ali Aouad, Adam N. Elmachtoub, Kris J. Ferreira, Ryan McNellis (2023) Market Segmentation Trees. Manufacturing & Service Operations Management 25(2):648-667. https://doi.org/10.1287/msom.2023.1195

## Motivation
1. However, such a market segmentation is driven only by user feature dissimilarity rather than differences in user response behavior, which
can be particularly problematic in cases where the key user features distinguishing market segments are not key drivers of response behavior.

## Model
1. MSTs take the structural form of model trees, which refer to a generalization of decision trees that allow for nonconstant leaf prediction models.
2. none of the previous methods select decision tree splits that directly minimize the predictive error of the resulting collection of leaf models
3. We argue that the approach of first clustering and then fitting response models within each cluster suffers from a fundamental limitation; namely, the resulting market segmentation does not take into account the predictive accuracy of the resulting collection of response models but is instead driven only by minimizing withincluster feature dissimilarity
4. MSTs also naturally model interactions among the contextual variables;