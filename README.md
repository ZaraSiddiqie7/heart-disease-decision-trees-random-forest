# Task 5: Decision Trees and Random Forests

Internship Task (Elevate Labs – Data Analyst Internship)

## Objective
Learn tree-based models for classification using the Heart Disease Dataset — training and visualizing a Decision Tree, controlling overfitting, comparing against a Random Forest, interpreting feature importance, and validating results with cross-validation.

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib (tree visualization via `plot_tree`)

## Dataset
Heart Disease Dataset — 13 clinical features (age, chest pain type, cholesterol, max heart rate, etc.) with a binary target indicating presence/absence of heart disease.

**Note:** The raw file contained 1025 rows with 723 duplicates. After removing duplicates, the working dataset has **302 unique patient records**. Deduplication was essential — without it, accuracy scores were artificially inflated (~99%) due to data leakage between train and test splits.

## Steps Performed
1. **Decision Tree Classifier** — trained a full-depth tree and visualized it with `plot_tree`.
2. **Overfitting Analysis** — swept `max_depth` from 1–20, comparing train vs. test accuracy to find the optimal depth (best depth = 3).
3. **Random Forest** — trained an ensemble of 200 trees and compared accuracy against the Decision Tree.
4. **Feature Importance** — extracted and visualized feature importances from the Random Forest.
5. **Cross-Validation** — evaluated both models using 5-fold cross-validation for a more robust accuracy estimate.

## Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree (full depth) | 1.000 | 0.803 |
| Decision Tree (max_depth=3) | 0.859 | 0.803 |
| Random Forest (200 trees) | 1.000 | 0.754 |

**5-Fold Cross-Validation:**
- Decision Tree (max_depth=3): mean = **0.795**
- Random Forest: mean = **0.821**

**Top Features by Importance (Random Forest):**
1. `cp` (chest pain type) — 16.6%
2. `thalach` (max heart rate achieved) — 12.9%
3. `ca` (number of major vessels) — 10.9%
4. `thal` (thalassemia result) — 10.0%
5. `oldpeak` (ST depression induced by exercise) — 9.3%

## Key Takeaways
- An unpruned Decision Tree overfits heavily (100% train vs. 80.3% test accuracy); limiting `max_depth` to 3 keeps test accuracy the same while greatly reducing overfitting and improving interpretability.
- On a single train/test split, the pruned Decision Tree slightly outperformed Random Forest — but 5-fold cross-validation (a more reliable estimate across multiple splits) shows **Random Forest generalizes better on average**, confirming the value of ensemble methods.
- Chest pain type, max heart rate, and vessel/thalassemia results are the strongest predictors of heart disease in this dataset — consistent with clinical knowledge.

## Files
- `task5_decision_trees_random_forests.ipynb` — full notebook with code, visualizations, and outputs
- `heart.csv` — dataset used
- `images/` — exported plots (decision tree, overfitting curve, model comparison, feature importance)

## What I Learned
Decision trees, ensemble learning (Random Forests), feature importance interpretation, overfitting control via hyperparameter tuning, and cross-validation for robust model evaluation.
