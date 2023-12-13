# BZAN 645: Final Project
## Introduction
In this project, we use the processed Petfinder data from the midterm project. Since the color and breed features are categorical, and one-hot encoding is not that informative, we use the embeddings of these features as the input of the model. Combining the embeddings, we get a better prediction outcome.

## Experiment
### Data Preprocessing
For simplicity, we use the same processed data from the midterm project for a better comparison. 

### Model
#### Embedding
We use a neural network to learn the embeddings of the categorical features. The process is as follows:
1. We set the embedding dimension to 5 or 10.
2. Combining the embeddings of the categorical features and the continuous features, we get the input of the model.
3. We use a neural network with multiple hidden layers to learn the embeddings by predicting the adoption speed.
4. Finally, we get the embeddings of the categorical features by extracting the weights of the embedding layers.

The neural network is defined as follows:
```python
PetFinderModel(
  (embeddings): ModuleList(
    (0): Embedding(176, 5)
    (1): Embedding(135, 5)
    (2): Embedding(3, 2)
    (3-4): 2 x Embedding(7, 4)
    (5): Embedding(6, 3)
    (6): Embedding(14, 5)
    (7): Embedding(812, 5)
    (8): Embedding(63, 5)
    (9): Embedding(142, 5)
  )
  (lin1): Linear(in_features=201, out_features=512, bias=True)
  (lin2): Linear(in_features=512, out_features=256, bias=True)
  (lin3): Linear(in_features=256, out_features=128, bias=True)
  (lin4): Linear(in_features=128, out_features=32, bias=True)
  (lin5): Linear(in_features=32, out_features=1, bias=True)
  (bn1): ReLU()
  (bn2): ReLU()
  (bn3): ReLU()
  (bn4): ReLU()
  (output): ReLU()
  (emb_drop): Dropout(p=0.2, inplace=False)
  (drops): Dropout(p=0.1, inplace=False)
)
```
The embedding layers correspond to the following features:
```python
cat_cols = [
    "Breed1",
    "Breed2",
    "Color1",
    "Color2",
    "Color3",
    "Gender",
    "State",
    "Breed_full",
    "Color_full",
    "hard_interaction",
]
```


#### XGBoost with Embedding
Combining the embeddings of the categorical features and the continuous features, we get the input of the XGBoost regression model. Then we use the XGBoost model to predict the adoption speed. We use the same parameters as the midterm project. By a 10-fold cross-validation, we get the following results:
| Embedding Dimension | Quadratic Weighted Kappa | Standard Deviation |
| --- | --- | --- |
|0|0.473238|0.01839972|
| 5 | 0.469962 | 0.02144644 |
| 10 | 0.475309 | 0.01566751 |

In this table, embedding dimension 0 means that we do not use the embeddings of the categorical features. We can see that the model with embedding dimension 10 has the best performance on both weighted kappa and standard deviation.

## Conclusion
In this project, we use the embeddings of the categorical features as the input of the XGBoost model. We find that the model with embedding dimension 10 has the best performance on both weighted kappa and standard deviation. The result shows that the embeddings of the categorical features are useful for the prediction of the adoption speed. We may also try a higher embedding dimension for some features to get a better result. For example, the hard_interaction feature has 812 unique values, and the embedding dimension is 5. We may try a higher embedding dimension for this feature.

Also, there might be some other ways to use the embeddings of the categorical features. For example, we can use the embeddings as the only input of a neural network to predict the adoption speed directly. This may be a good way to use the embeddings of the categorical features and avoid the problem of combining the embeddings and the continuous features.

The code is in https://github.com/Jinyi-Liu/BZAN645/tree/main/Final.