<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-faders/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  FaDeRS: Fairness and Depolarization in Recommender Systems
</h3>

<p align="center">WebMedia 2025: 31st Brazilian Symposium on Multimedia and the Web</p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-faders?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-faders/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-faders?style=social">
  </a>
</p>

## :page_with_curl: About the project <a name="-sobre"/></a>

Context: Recommender systems have become fundamental in modern society as they help navigate the vast amount of available data, enabling users to find information, products, or services more efficiently and personally. They directly impact how people consume data, goods, and resources. Problem: Recommender systems often lack fairness and diversity, resulting in unfair services and increased preference polarization. Solution: This work presents FaDeRS (Fairness and Depolarization in Recommender Systems), an approach aimed at increasing fairness and diversity in recommender systems. FaDeRS adjusts predictions through controlled perturbations and optimization to mitigate individual unfairness and polarization without modifying the input data. IS Theory: The research is related to socio-technical theory, addressing one of the socio-algorithmic problems, algorithmic discrimination. We consider a specific set of approaches to encode fair behaviors. Method: The research applied a quantitative method with experimentation using two datasets in distinct contexts, implementing a post-processing algorithm based on the Simulated Annealing meta-heuristic. Summary of Results: The proposed framework demonstrated significant reductions in polarization (up to 78.64\%) and individual unfairness (up to 33.97\%), with only a small increase in the Root Mean Square Error (RMSE), indicating an improvement in the socially desirable qualities of the systems without unduly sacrificing accuracy. Contributions and Impact in the IS field: The main contribution is a mechanism that balances personalization and fairness, simultaneously addressing polarization and individual unfairness from the items' perspective, promoting a fairer and more diverse approach to recommendation.

### :balance_scale: Fairness Measures <a name="-medidas"/></a>

* **Polarization**: To capture polarization, we seek to measure the extent to which user ratings disagree. Thus, to measure user polarization, we consider the estimated ratings `$\hat{X}$`, and define the polarization metric as the normalized sum of the Euclidean distances between pairs of estimated user ratings, that is, between rows of `$\hat{X}$`.

* **Individual fairness**: For each item `$j$`, we define `$\ell_j$`, the loss of item `$j$`, as the mean squared error of the estimate over the known ratings of item `$j$`.

### :chart_with_upwards_trend: Results <a name="-resultados"/></a>

[Link to the Excel file](https://github.com/ravarmes/recsys-depolarize-fair/blob/main/_results-article.xlsx)

### Files

| File                                 | Description                                                                                                                                                                                                                                         |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmDepolarizeFair             | Class to implement fairness and depolarization of recommendations from recommendation system algorithms.                                                                                                                                           |
| AlgorithmUserFairness               | Classes to measure fairness (polarization, individual fairness, and group fairness) of recommendations from recommendation system algorithms.                                                                                                         |
| RecSys                               | Factory pattern class to instantiate a recommendation system based on string parameters.                                                                                                                                                           |
| RecSysALS                            | Alternating Least Squares (ALS) for Collaborative Filtering is an algorithm that iteratively optimizes two matrices to better predict user ratings on items, based on the idea of matrix factorization.                                              |
| RecSysKNN                            | K-Nearest Neighbors for Recommendation Systems is a method that recommends items or users based on the proximity or similarity between them, using the technique of the K nearest neighbors.                                                          |
| RecSysNMF                            | Non-Negative Matrix Factorization for Recommendation Systems uses the decomposition of a ratings matrix into two non-negative factor matrices, revealing latent patterns that can be used to predict missing ratings.                               |
| RecSysSGD                            | Stochastic Gradient Descent for Recommendation Systems is an optimization technique that iteratively adjusts the model parameters to minimize prediction error in ratings, through updates based on gradients calculated stochastically.            |
| RecSysSVD                            | Singular Value Decomposition for Recommendation Systems is a method that factors the ratings matrix into three smaller matrices, capturing essential information about users and items, which facilitates recommendations by reconstructing the original matrix with missing data filled in. |
| RecSysNCF                            | Neural Collaborative Filtering is a modern approach to collaborative filtering that uses neural networks to model complex, nonlinear interactions between users and items, aiming to enhance recommendation quality.                                    |
| TestAlgorithmDepolarizeFair         | Test script for the depolarization and fairness algorithm (AlgorithmDepolarizeFair) |

## :hammer_and_wrench: Usage

- Main class: src/TestAlgorithmDepolarizeFair.py — just run this script to execute the FaDeRS algorithm end-to-end.
- Install dependencies:
  - Windows: `python -m pip install -r requirements.txt`
- Run the experiments:
  - `python src/TestAlgorithmDepolarizeFair.py`
- Minimal example:
  - Create an estimated ratings matrix using one of the RecSys algorithms and then apply the AlgorithmDepolarizeFair to reduce polarization and individual unfairness while keeping accuracy acceptable.

```python
from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, RMSE
from AlgorithmDepolarizeFair import AlgorithmDepolarize

# Read dataset and estimate ratings
recsys = RecSys(n_users=1000, n_items=1000, top_users=True, top_items=True)
X, users_info, items_info = recsys.read_dataset(1000, 1000, True, True, data_dir="data/MovieLens-1M")
omega = ~X.isnull()
X_est = recsys.compute_X_est(X, algorithm='RecSysSVD')

# Evaluate fairness/accuracy
pol = Polarization(); ilv = IndividualLossVariance(X, omega, 0); rmse = RMSE(X, omega)
print(pol.evaluate(X_est), ilv.evaluate(X_est), rmse.evaluate(X_est))

# Apply depolarization/impartiality algorithm
alg = AlgorithmDepolarize(X, omega, 0)
list_X_est = alg.evaluate(X_est, alpha=0.4, h=20)

# Build optimization matrices and final recommendation
losses = [ilv.get_losses(m) for m in list_X_est]
pols = [pol.get_polarizations(m) for m in list_X_est]
ZIL = AlgorithmDepolarize.losses_to_ZIL(losses, n_items=X.shape[1])
ZIP = AlgorithmDepolarize.polarizations_to_ZIP(pols, n_items=X.shape[1])
X_pi = AlgorithmDepolarize.make_matrix_X_pi_annealing(list_X_est, ZIL, ZIP)
```

Notes:
- Excel export requires the openpyxl package (already listed in requirements.txt).
- If you plan to use RecSysNCF, install TensorFlow separately and ensure a compatible version.

## :email: Contact

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**
