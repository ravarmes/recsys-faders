import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt

class RecSysSVD:
    def __init__(self, n_factors=50, ratings=None):
        # Initialize SVD recommender with number of latent factors.
        self.n_factors = n_factors
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        # Set ratings and fill NaNs for SVD compatibility.
        self.ratings = ratings.fillna(0)  # Fill missing values with 0 for SVD
        # Note: Filling with mean or other value may be considered depending on the use case

    def fit_model(self):
        # Factorize ratings via SVD and compute predictions and RMSE.
        # Get the values from the ratings matrix
        matrix = self.ratings.values
        
        # Apply SVD to the ratings matrix
        U, sigma, Vt = svds(matrix, k=self.n_factors)
        
        # Transform sigma into a diagonal matrix
        sigma = np.diag(sigma)
        
        # Compute approximate ratings using the factor matrices
        approx_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Convert the approximate ratings back to a DataFrame
        self.predictions = pd.DataFrame(approx_ratings, index=self.ratings.index, columns=self.ratings.columns)
        
        # Compute RMSE using only the original non-null ratings
        mask = self.ratings > 0
        mse = mean_squared_error(self.ratings[mask], self.predictions[mask])
        rmse = sqrt(mse)
        
        return self.predictions, rmse

    def recommend_items(self, user_id, top_n=10):
        # Return top-N not-yet-rated items for a user.
        # Get predictions for a specific user
        user_predictions = self.predictions.loc[user_id].sort_values(ascending=False)
        
        # Get items already rated by the user
        known_items = self.ratings.loc[user_id] > 0
        
        # Filter recommendations to include only items not yet rated by the user
        recommendations = user_predictions[~known_items].head(top_n)
        
        return recommendations

# Usage example:
# Assuming `ratings_df` is your ratings DataFrame with users as rows and items as columns, with NaN for missing ratings
# ratings_df = pd.DataFrame(...)
# rec_sys_svd = RecSysSVD(n_factors=50, ratings=ratings_df)
# predictions, rmse = rec_sys_svd.fit_model()
# print(f"RMSE: {rmse}")
# recommendations = rec_sys_svd.recommend_items(user_id='some_user_id', top_n=10)
# print(recommendations)
