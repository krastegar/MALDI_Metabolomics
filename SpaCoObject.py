# ============================================================================
# SPACO Class Overview
# ============================================================================
#
# Description:
# -----------
# The SPACO class is designed for Spectral Filtering and Projection using
# Principal Components Analysis (PCA) and Graph Laplacian. It takes in sample
# features and a neighbor matrix as inputs and provides methods for
# preprocessing, PCA whitening, spectral filtering, and projection.

# Methods
# -------
#
# The following methods are available in the SPACO class:
#
# 1. `__init__`: Initializes a SPACO object with sample features, neighbor matrix,
#    and optional parameters for PCA and spectral filtering.
#
# 2. `preprocess`: Preprocesses the sample features array by ensuring it is in
#    the correct shape, removing constant features, and scaling using StandardScaler.
#
# 3. `pca_whitening`: Performs PCA whitening on the sample features array and
#    returns the whitened data. Optional parameters include the threshold (c) for
#    selecting the variance of the principal components
#
# 4. `_resample_lambda_cut`: Resamples the eigenvalues of the graph Laplacian
#    matrix to estimate the eigenvalue threshold for spectral filtering. Uses
#    percentile and resample_iterations as inputs.
#
# 5. `spectral_filtering`: Performs spectral filtering on the whitened data using
#    the estimated eigenvalue threshold and returns the filtered eigenvalues and
#    eigenvectors.
#
# 6. `spaco_projection`: Projects the original data onto the filtered eigenvectors
#    and returns the projected data.
#
# 7. `spaco_test`: Computes a test statistic for a given input vector x using the
#    projected data and graph Laplacian.
#
# ============================================================================
# Date: 20/02/2025
# Author: Kiarash Rastegar
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import igraph as ig
import leidenalg as la
import plotly.express as px
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from scipy.stats import t
from scipy.sparse import csr_matrix
from scipy import integrate
from typing import Tuple
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


class SPACO:
    def __init__(
        self,
        sample_features,
        coords,
        neighbormatrix=None,
        c=0.95,
        compute_nSpacs=True,
        percentile=95,
    ):
        """
        Initialize a SpaCo object.

        Parameters
        ----------
        sample_features : array-like (n_samples, n_features)
            The sample features array.
        neighbormatrix : array-like (n_samples, n_samples)
            The neighbor matrix.
        coords (numpy.ndarray):
                Array of shape (n_samples, 2) containing the x and y coordinates for each sample point.
        c : float, optional
            The threshold for selecting the minimal number of principal components. Default is 0.95.
        percentile : int, optional
            The percentile of the eigenvalues that is used to set the eigenvalue threshold. Default is 95.

        Returns
        -------
        None
        """
        self.percentile: int = percentile
        self.compute_nSpacs: bool = compute_nSpacs
        self.SF: np.ndarray = self.__preprocess(sample_features)
        self.coords: np.ndarray = self.__rotate_coordinates(coords)
        self.A: np.ndarray = self.__generate_adjacency_matrix(self.coords) if neighbormatrix is None else self.__generate_adjacency_matrix(self.coords, neighbormatrix)
        self.c: float = c
        self.graphLaplacian = np.asarray((1 / self.A.shape[0]) * np.eye(self.A.shape[0]) + (
            1 / np.abs(self.A).sum()
        ) * self.A)
        # lazy loading variables 
        self.sample_names = None
        self.feature_names = None
        self.lambda_cut = None
        self.nSpacs = None
        self.loadings = None # to store loadings after projection
        self._cache = {}

    def __getattr__(self, name):
        """
        Lazy loading of attributes. Computes the attribute if it is not already cached.

        The cache is stored in the `_cache` attribute of the object.

        The possible attributes that can be lazily loaded are:
        - whitened_data
        - spectral_results
        - graphLaplacian
        - sampled_sorted_eigvecs
        - sampled_sorted_eigvals
        - Pspac
        - Vk
        - sigma
        - sigma_eigh
        - non_random_eigvals
        - results_all

        If the attribute is not one of the above, an AttributeError is raised.
        """
        if name in self._cache:
            # If the attribute is already cached, return it
            return self._cache[name]

        # Compute the attribute if it is not already cached
        if name == "whitened_data":
            # Compute the whitened data using PCA
            self._cache[name] = self.__pca_whitening()
        elif name == "spectral_results":
            # Compute the spectral results using the graph Laplacian
            self._cache[name] = self.__spectral_filtering()
        elif name == "graphLaplacian":
            # Compute the graph Laplacian
            self._cache[name] = self.spectral_results[2]
        elif name == "sampled_sorted_eigvecs":
            # Compute the sorted eigenvectors of the spectral results
            self._cache[name] = self.spectral_results[0]
        elif name == "sampled_sorted_eigvals":
            # Compute the sorted eigenvalues of the spectral results
            self._cache[name] = self.spectral_results[1]
        elif name == "Pspac" or name == "Vk":
            # Compute the projection of the sample features onto the SPACO space
            self._cache["Pspac"], self._cache["Vk"] = self.spaco_projection()
        elif name == "sigma" or name == "sigma_eigh":
            # Compute the eigenvalues of the graph Laplacian
            self._cache["sigma_eigh"], self._cache["sigma"] = self.__sigma_eigenvalues()
        elif name == "non_random_eigvals":
            # Cache the non-random eigenvalues
            return self.non_random_eigvals
        elif name == "results_all":
            # Cache the results of the shuffle decomposition
            return self.results_all
        else:
            # Raise an AttributeError if the attribute is not one of the above
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        return self._cache[name]

    def __remove_constant_features(self, X: np.ndarray) -> np.ndarray:
        """
        Remove constant features from the data using variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            The input data with constant features removed.

        Raises
        ------
        ValueError
            If all features are constant, a ValueError is raised.
        """
        # Remove constant features
        X = X[:, np.var(X, axis=0) > 0]

        if np.all(X == 0):
            raise ValueError("No features left after removing constant features.")
        return X

    def __preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the sample features array. normalize and remove constant features
        """
        # if the sample feature matrix is a pandas dataframe we capture the feature and sample names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.sample_names = X.index.tolist()
            X = X.to_numpy()

        # We want to see if the numpy array is better for calculation or csr matrix based on sparsity
        sparsity = 1 - np.count_nonzero(X) / X.size
        if sparsity > 0.9:
            X = csr_matrix(X)
            # need to z-scale for csr matrix (so we perserve sparsity)
            return 
        else: 
            scaler = StandardScaler()
            X = self.__remove_constant_features(X)
            return scaler.fit_transform(X)

    def __generate_adjacency_matrix(self, coords: np.ndarray, n_neighbors: int = 10, neighbor_matrix: np.ndarray = None) -> np.ndarray:


        if neighbor_matrix is not None:
            sparsity = 1- np.count_nonzero(neighbor_matrix) / neighbor_matrix.size
            if sparsity > 0.9:
                return csr_matrix(neighbor_matrix)
            else: 
                return np.asarray(neighbor_matrix)


        # Check if the coordinates are provided as a pandas DataFrame
        if not isinstance(coords, pd.DataFrame):
            raise ValueError("Coordinates must be provided as a pandas DataFrame.")
        
        # Run K-Nearest Neighbors to generate adjacency matrix
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', n_jobs=-1)
        knn.fit(coords)
        adjacency_matrix = knn.kneighbors_graph(coords, n_neighbors=n_neighbors, mode='distance')

        return adjacency_matrix


    def __orthogonalize(
        self,
        V: np.ndarray,
        L: np.ndarray,
        tol: float = np.sqrt(np.finfo(float).eps),
    ) -> np.ndarray:
        """
        Perform Gram-Schmidt orthogonalization under the L-inner product.

        Parameters:
        - V: (n, k) metapatterns found in L-space
        - L: (n, n) graph Laplacian
        - tol: tolerance to skip near-null vectors under L

        Returns:
        - Q: (n, r) L-orthonormal basis, where r <= k

        Raises:
        - ValueError: if all vectors are rejected (i.e., null in L-space)
        """
        _, k = V.shape
        Q = []
        L = np.asarray(L)
        for i in range(k):
            v = V[:, i].copy()  # make a copy to avoid modifying the original vector
            for q in Q:
                proj_coeff = q.T @ L @ v
                v -= proj_coeff * q

            norm_L = np.sqrt(v.T @ L @ v)
            if norm_L < tol:
                continue

            q = v / norm_L
            Q.append(q)

        if not Q:
            raise ValueError(
                "All vectors are null or near-null under the L-inner product. No basis vectors remain."
            )

        Q = np.stack(Q, axis=1)
        return Q

    def __pca_whitening(self) -> np.ndarray:
        """
        Perform PCA whitening on the input data.

        This whitening process is done using the eigenvectors and eigenvalues of
        the covariance matrix of the input data.

        The eigenvectors are sorted in descending order of the corresponding
        eigenvalues. The top r eigenvectors and eigenvalues are then selected to
        form a new matrix W_r and diagonal matrix D_r, respectively.

        The input data is then projected onto the column space of W_r and scaled
        by the inverse square root of D_r to obtain the whitened data.
        """
        # Compute the covariance matrix of the input data
        cov = np.cov(self.SF, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute the total variance and cumulative variance
        total_variance = np.sum(eigenvalues)
        cumulative_variance = np.cumsum(eigenvalues)

        # Find the index of the top r eigenvectors that capture the specified
        # proportion of variance (self.c)
        r = np.searchsorted(cumulative_variance / total_variance, self.c) + 1

        # Select the top r eigenvectors and eigenvalues
        W_r = eigenvectors[:, :r]
        D_r = np.diag(eigenvalues[:r])

        # Compute the inverse square root of D_r
        D_r_inv_sqrt = np.linalg.inv(np.sqrt(D_r))

        # Project the input data onto the column space of W_r and scale by
        # the inverse square root of D_r
        whitened_data = np.dot(self.SF, W_r).dot(D_r_inv_sqrt)

        return whitened_data

    def __shuffle_decomp(self) -> float:
        """
        Shuffles the rows of the input matrix X and computes the largest eigenvalue
        of the shuffled matrix using the eigs function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        largest_eigenvalue : float
            The largest eigenvalue of the shuffled matrix.
        """

        # Shuffle rows using indices for efficiency
        indices = np.random.permutation(self.whitened_data.shape[0])
        shuffle_reduced_data = self.whitened_data[indices, :]

        # Compute the matrix M which is the product of the whitened data and the graph Laplacian
        # M is a symmetric matrix
        M = shuffle_reduced_data.T @ self.graphLaplacian @ shuffle_reduced_data

        # Compute the largest eigenvalue of M

        # eigs returns the eigenvalues and eigenvectors of M
        # We only need the largest eigenvalue so we set k=1
        # The eigenvectors are not needed so we set which="LR"
        largest_eigenvalue = eigs(M, k=1, which="LR", tol=1e-4)[0][0].real

        return float(largest_eigenvalue)

    def __CI_SE(self, results_all: list[float]) -> Tuple[float, float]:
        """
        Compute the 95% confidence interval and standard error of the mean for a list of values.

        Parameters
        ----------
        results_all : List[float]
            A list of values to compute the confidence interval and standard error for.

        Returns
        -------
        ci_lower : float
            The lower bound of the 95% confidence interval.
        ci_upper : float
            The upper bound of the 95% confidence interval.
        """
        # Calculate the mean of the results
        mean = np.mean(results_all)

        # Calculate the standard error of the mean
        # np.std calculates the standard deviation of the list
        # Divide by the square root of the number of observations
        std_error: float = np.std(results_all) / np.sqrt(len(results_all))

        # Determine the t critical value for 95% confidence
        # t.ppf gives the value of the t-distribution for a given cumulative probability
        # 0.975 is used to find the two-tailed critical value for 95% confidence
        # df is degrees of freedom, which is number of observations minus one
        t_critical = t.ppf(0.975, df=len(results_all) - 1)

        # Calculate the margin of error
        # This is the product of the t critical value and the standard error
        margin_of_error = t_critical * std_error

        # Calculate the lower and upper bounds of the confidence interval
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error

        return float(ci_lower), float(ci_upper)

    def __replicate(self, n_replicates: int) -> np.ndarray:
        """
        Replicate the __shuffle_decomp method n_iterations times using multithreading.

        This method takes in the number of replicates to perform and uses
        multithreading to perform the __shuffle_decomp method in parallel.

        This is done by creating a ThreadPoolExecutor object, which is a context
        manager that represents a pool of worker threads. The map method of the
        executor is then used to apply the __shuffle_decomp method to each element
        of an iterable (in this case, a range object from 0 to n_replicates-1).

        The results of each call are collected in a list, which is then converted
        to a numpy array and returned.

        Parameters
        ----------
        n_replicates : int
            The number of times to replicate the __shuffle_decomp method.

        Returns
        -------
        results_all : np.ndarray
            A numpy array containing the results of each replicate.
        """
        with ThreadPoolExecutor() as executor:
            # Create an iterable from 0 to n_replicates-1
            iterable = range(n_replicates)

            # Use the map method of the executor to apply the __shuffle_decomp method
            # to each element of the iterable
            results_all = list(
                executor.map(lambda _: self.__shuffle_decomp(), iterable)
            )

            # Convert the list of results to a numpy array
            results_all = np.array(results_all)

            # Return the numpy array
            return results_all

    @lru_cache(maxsize=None)
    def __resample_lambda_cut(
        self,
        non_random_eigvals: np.ndarray,
        batch_size: int = 50,
        n_replicates: int = 100,
        n_simulations: int = 1000,  # checking with a lower number of iterations
    ) -> float:
        """
        Resamples the shuffled adjacency matrix to calculate the confidence interval for the relevant number of SpaCs.
        Generating a confidence interval from the shuffled M matrix eigenvalues representing random noise. This is done
        by using the replicate function to run the __shuffle_decomp method multiple times. The CI is then used to determine the relevant number of SpaCs.
        The method iteratively decreases the confidence interval until the number of eigenvalues within the
        confidence interval is 1 or the number of iterations exceeds n_simulations.

        Parameters
        ----------
        batch_size : int, optional (default=10)
            The number of times to replicate the __shuffle_decomp method in each iteration.
        n_iterations : int, optional (default=100)
            The number of times to replicate the __shuffle_decomp method.
        n_simulations : int, optional (default=1000)
            The maximum number of iterations to perform.

        Returns
        -------
        rel_spacs_idx : int
            The relevant number of SpaCs.
        """
        # need to remake non_random_eigvals to be a numpy array for filtering
        non_random_eigvals = np.array(non_random_eigvals)

        # shuffle / permute the neighbor matrix
        results_all: np.ndarray = self.__replicate(n_replicates)

        # Caching the initial results of the shuffle decomposition (
        # ie. the eigenvalues of the whitened data that have been rotated in SPACO space
        self._cache["non_random_eigvals"] = non_random_eigvals

        # calculate the 95 CI and SE
        ci_lower, ci_upper = self.__CI_SE(list(results_all))

        # Select the eigenvalues from results_all that are within the 95% CI
        # (i.e. the eigenvalues that are not significantly different from the null hypothesis)
        lambdas_inCI = non_random_eigvals[
            (non_random_eigvals >= ci_lower) & (non_random_eigvals <= ci_upper)
        ]

        iterations: int = 0

        # if there are no lambdas in the CI, we return the upper bound of the CI
        # this is the case when the CI is too small and the lambdas are not in the CI
        if len(lambdas_inCI) == 0:
            # Also going to cache the initial results of the shuffle decomposition
            # i.e)  the eigenvalues of the randomly permuted whitened data that has
            #       then been rotated in SPACO space
            self._cache["results_all"] = results_all
            self.lambda_cut = ci_upper
            return self.lambda_cut

        # If there is only one lambda in the CI, we can return it
        if len(lambdas_inCI) == 1:
            # lambdas_inCI should be a 1D array with only one element, which is the lambda of interest
            self.lambda_cut = lambdas_inCI[0]
            self._cache["results_all"] = results_all
            return self.lambda_cut

        # if there are more than 1 lambdas in the CI, we need to iterate
        # until there is only 1 lambda in the CI or the number of iterations is greater than n_simulations
        while len(lambdas_inCI) > 1:
            iterations += 1
            # Adding the batch size to the number of iterations to steadily decrease the CI margins
            batch_results = self.__replicate(batch_size)

            # Calculate the 95% CI and SE for the new batch of results
            results_all = np.append(results_all, batch_results)

            # Caching the results of the shuffle decomposition
            self._cache["results_all"] = results_all

            # Calculate the 95% CI and SE for the new batch of results
            ci_lower, ci_upper = self.__CI_SE(list(results_all))

            # checking to see how many lambdas are in the CI
            lambdas_inCI = non_random_eigvals[
                (non_random_eigvals >= ci_lower) & (non_random_eigvals <= ci_upper)
            ]

            if len(lambdas_inCI) < 2:
                # lambdas_inCI should be a 1D array with only one element, which is the lambda of interest
                self.lambda_cut = (
                    lambdas_inCI[::-1][0] if len(lambdas_inCI) > 0 else ci_upper
                )

                self._cache["results_all"] = results_all
                return self.lambda_cut

            if iterations >= n_simulations:
                print(
                    f"Reached maximum number of iterations: {n_simulations}.\n # of elements in CI: {len(lambdas_inCI)}"
                )
                # if ther are still more than 1 eigenvalue in the CI, we take the upper bound of the CI
                self._cache["results_all"] = results_all
                self.lambda_cut = ci_upper
                return self.lambda_cut

        # Ensure a float is always returned
        # If the loop exits without returning, return the current lambda_cut or ci_upper as fallback
        if self.lambda_cut is not None:
            self._cache["results_all"] = results_all
            return self.lambda_cut
        else:
            return float(ci_upper)

    def __spectral_filtering(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform spectral filtering on the whitened data using the graph Laplacian.

        This is the core of the SPACO algorithm. We take the whitened data, which is the
        sample features after PCA whitening, and project it onto the graph Laplacian.

        The graph Laplacian is a symmetric matrix that represents the graph structure of
        the data. It is defined as the sum of the neighbor matrix and the diagonal matrix
        of node degrees. The graph Laplacian is then normalized by the sum of the node
        degrees.

        We then compute the eigenvectors and eigenvalues of the graph Laplacian.
        The eigenvectors are the principal components of the graph Laplacian, and the
        eigenvalues are the corresponding eigenvalues.

        The eigenvectors are sorted in descending order of their corresponding
        eigenvalues.

        If compute_nSpacs is True, we then compute the lambda cut, which is the
        threshold for selecting the number of eigenvectors to keep. The lambda cut is
        computed by taking the median of the eigenvalues and then resampling from the
        distribution of eigenvalues until the lambda cut is greater than or equal to
        the 95% CI of the resampled eigenvalues. This is done to get the largest cutoff
        of eigenvalues that are significantly different from a random distribution.

        If compute_nSpacs is False, the lambda cut is simply set to the median of the
        eigenvalues.

        The eigenvectors and eigenvalues are then filtered by the lambda cut to select
        the top k eigenvectors and eigenvalues.

        The output is a tuple of three arrays: the filtered eigenvectors, the filtered
        eigenvalues, and the graph Laplacian.
        """

        # Compute the graph Laplacian
        # The graph Laplacian is a symmetric matrix that represents the graph structure of
        # the data. It is defined as the sum of the neighbor matrix and the diagonal matrix
        # of node degrees. The graph Laplacian is then normalized by the sum of the node
        # degrees.

        # Compute the eigenvectors and eigenvalues of the graph Laplacian
        M = self.whitened_data.T @ self.graphLaplacian @ self.whitened_data
        eigvals, eigvecs = eigh(M)

        # Sort the eigenvectors in descending order of their corresponding eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        if self.compute_nSpacs:
            # Compute the lambda cut using resampling
            self.lambda_cut = self.__resample_lambda_cut(
                tuple(eigvals.tolist())
            )  # make a hashable data structure for cacheing
        else:
            # Compute the lambda cut using the median of the eigenvalues
            self.lambda_cut = np.median(eigvals)
        # Select the top k eigenvectors and eigenvalues
        k = len(eigvals[eigvals >= self.lambda_cut])
        sampled_sorted_eigvecs = eigvecs[:, :k]
        sampled_sorted_eigvals = eigvals[:k]
        return sampled_sorted_eigvecs, sampled_sorted_eigvals, self.graphLaplacian

    def spaco_projection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the sample features onto the SPACO space.

        Returns
        -------
        Pspac : np.ndarray
            The projection of the sample features onto the SPACO space.
        Vk : np.ndarray
            The matrix of SPACO projections.
        """
        # Compute the orthonormal basis U by dividing each eigenvector by the square root of its eigenvalue
        self.loadings = self.sampled_sorted_eigvecs / np.sqrt(self.sampled_sorted_eigvals)

        # Project the whitened data onto the orthonormal basis U to obtain the matrix Vk
        Vk = self.whitened_data @ self.loadings

        # Calculate the projection of the sample features onto the SPACO space
        # This involves projecting Vk, then transforming by the graph Laplacian, and finally projecting back
        # to the original sample feature space
        Pspac = Vk @ Vk.T @ self.graphLaplacian @ self.SF

        # Store the number of SPACO components
        self.nSpacs = Vk.shape[1]

        # Return both the projection into SPACO space and the matrix of SPACO projections
        return Pspac, Vk

    def __sigma_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigenvalues of the matrix sigma, which is a k x k matrix
        representing the weighted sum of the squared eigenvalues of the graph Laplacian
        matrix.

        Parameters
        ----------
        None

        Returns
        -------
        sigma_eigh : np.ndarray
            The eigenvalues of the sigma matrix.
        sigma : np.ndarray
            The sigma matrix itself.
        """

        # Compute the orthogonalized (unitary) matrix projection
        # This is done by taking the eigenvectors of the graph Laplacian
        # and dividing each eigenvector by the square root of its eigenvalue
        projection = self.__orthogonalize(V=self.Vk, L=self.graphLaplacian)

        # Sk is the orthogonalized matrix of size k x n
        Sk = projection[:, : self.Vk.shape[1]]

        # sigma is the k x k matrix representing the weighted sum of the squared
        # eigenvalues of the graph Laplacian matrix
        sigma = Sk.T @ self.graphLaplacian @ self.graphLaplacian @ Sk

        # Compute the eigenvalues of the sigma matrix
        sigma_eigh = np.linalg.eigvalsh(sigma)

        # Return both the eigenvalues and the sigma matrix
        return sigma_eigh, sigma


    @staticmethod
    def imhof(q, weights, dfs=None, ncs=None, sigma=0, lim=1000, acc=1e-6):
        """
        Compute the CDF of a weighted sum of chi-squared random variables using Imhof's method.
        
        Calculates P(Q > q) where Q = sum(weights[i] * X_i) + sigma * Z
        X_i ~ chi-squared(dfs[i], ncs[i]) and Z ~ N(0,1)
        
        Parameters
        ----------
        q : float
            The quantile at which to evaluate the CDF
        weights : array-like
            Weights for each chi-squared variable
        dfs : array-like, optional
            Degrees of freedom for each chi-squared (default: all 1)
        ncs : array-like, optional
            Non-centrality parameters (default: all 0, central chi-squared)
        sigma : float, optional
            Standard deviation of normal component (default: 0)
        lim : float, optional
            Upper limit for integration (default: 10000)
        acc : float, optional
            Absolute accuracy for integration (default: 1e-6)
        
        Returns
        -------
        float
            P(Q > q), the survival function value
        
        References
        ----------
        Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal variables.
        Biometrika, 48(3/4), 419-426.
        """
        weights = np.asarray(weights, dtype=np.float64)
        n = len(weights)
        
        if dfs is None:
            dfs = np.ones(n, dtype=np.float64)
        else:
            dfs = np.asarray(dfs, dtype=np.float64)
        
        if ncs is None:
            ncs = np.zeros(n, dtype=np.float64)
        else:
            ncs = np.asarray(ncs, dtype=np.float64)
        
        # Precompute to avoid repeated calculations
        sigma2 = sigma * sigma
        
        def integrand(u):
            """Integrand for Imhof's method"""
            if u == 0:
                return 0
            
            # Compute theta(u)
            theta = 0.5 * np.sum(dfs * np.arctan(weights * u))
            theta += 0.5 * np.sum(ncs * weights * u / (1 + weights**2 * u**2))
            theta -= 0.5 * q * u
            
            # Compute rho(u)
            log_rho = -0.25 * np.sum(dfs * np.log(1 + weights**2 * u**2))
            log_rho -= 0.5 * np.sum(ncs * weights**2 * u**2 / (1 + weights**2 * u**2))
            log_rho -= 0.5 * sigma2 * u**2
            
            rho = np.exp(log_rho)
            
            # Return the integrand: sin(theta) / u * rho
            return np.sin(theta) / u * rho
        
        # Perform numerical integration
        result, error = integrate.quad(integrand, 0, lim, 
                                    epsabs=acc, epsrel=acc,
                                    limit=10000)
        
        # Compute the survival function P(Q > q)
        prob = 0.5 - result / np.pi
        
        # Clamp to [0, 1] to handle numerical errors
        return np.clip(prob, 0, 1)
    
    def spaco_test(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Perform a statistical test in the SPACO framework.

        This method scales and normalizes the input vector `x`, computes a test statistic,
        and evaluates its significance against the distribution of eigenvalues of the
        graph Laplacian. The test statistic is computed as the squared projection of the
        normalized `x` onto the SPACO space, represented by the matrix `Vk`.

        Parameters
        ----------
        x : np.ndarray
            The input vector to be tested. It should be a 1D array representing sample features.

        Returns
        -------
        Tuple[float, float]
            The p-value of the test statistic and the test statistic itself.

        Raises
        ------
        AssertionError
            If the input vector `x` is not centered and scaled correctly.
        """
        gene : np.ndarray = np.asarray(x)
        if not np.isclose(gene.mean(), 0, atol=1e-4) and np.isclose(
            gene.std(), 1, atol=1e-4
        ):
            print("Gene is not centered and scaled\nProceeding to centering and scaling...")
            gene: np.ndarray = (gene - gene.mean()) / gene.std()

        # Compute the eigenvalues of the transformed matrix and the graph Laplacian (L)
        sorted_sigma_eigh = self.sigma_eigh[::-1]
        # printing all the variables berfore the test statistic

        # Normalize the scaled data
        gene = gene / np.repeat(np.sqrt(gene.T @ self.graphLaplacian @ gene), len(gene))
        # print(f"sigma: {self.sigma.shape}\ngene: {gene.shape}")

        # Compute the test statistic
        # calculate the statistic in steps
        Vk_proj_x: np.ndarray = gene.T @ self.graphLaplacian @ self.Vk
        test_statistic: float = float(Vk_proj_x.T @ Vk_proj_x)

        # print(f'test statistic: {test_statistic}\n\n\n sorted_sigma_eigenvals: {sorted_sigma_eigh[:nSpacs]}')
        # pval test statistic
        pVal = self.imhof(test_statistic, sorted_sigma_eigh)

        return pVal, test_statistic
    
    def features_in_spacs(
            self,
            n_neighbors,  
            clustering='leiden', 
            normalization=True, 
            plot=True, 
            save_file=False,
            k=10):
        
        # filtering for significant features only 
        # run tests, extract p-values, invert them, build mask, filter columns
        print("Testing for significant features...")
        pvals = [1 - self.spaco_test(self.SF[:, col])[0] for col in range(self.SF.shape[1])]
        mask = [p < 0.05 for p in pvals]
        significant_feature_filtered = self.SF[:, mask]  
        print(significant_feature_filtered.shape[1], " significant features identified out of ", self.SF.shape[1], " total features. ")
        # in case the inputted data is just given as a numpy.array
        if self.feature_names is None:
            self.feature_names = [i for i in range(self.SF.shape[1])]# at least label the columns whatever they are 
            self.feature_names = [feature for i, feature in enumerate(self.feature_names) if mask[i]]# significant feature colums saved 

        else:
            feature_names = self.feature_names.tolist()
            self.feature_names = [feature for i, feature in enumerate(feature_names) if mask[i]]

        # check to see if we filtered out any of the non-significant features    
        if significant_feature_filtered.shape[1] == self.SF.shape[1]:
            for i in range(5):
                print(" (-_-)/  ALL OF THE FEATURES HAVE BEEN TESTED AS SIGNIFICANT ")
            

        # lets go with the next steps in the clustering 
        # embedding the features in meta-pattern space 
        # center and scaling the data
        scaler = StandardScaler()
        # correlation matrix between features and meta-patterns
        # 1) center columns (features and meta-patterns)
        SFc = scaler.fit_transform(significant_feature_filtered)     # n x p
        MPc = scaler.fit_transform(self._cache['Vk']) # n x k

        # 2) dot product matrix (p x k)
        #C_num = SFc.T @ MPc   # p x k # 
        C_num = SFc.T @ self.graphLaplacian @ MPc 

        if normalization:
            C_num = normalize(C_num, axis=0, norm='l2') # feature/ column normalization
        
        # clustering algorithm
        if clustering == 'leiden':
            # ---------------------------------------------------------
            # 1. Compute cosine KNN graph
            # ---------------------------------------------------------

            # X: your data matrix (n_samples × n_features)
            # choose your neighborhood size

            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="cosine"
            ).fit(C_num)

            distances, indices = nbrs.kneighbors(C_num)

            # ---------------------------------------------------------
            # 2. Build igraph graph with cosine similarity weights
            # ---------------------------------------------------------
            edges = []
            weights = []

            for i in range(indices.shape[0]):
                for j, d in zip(indices[i], distances[i]):
                    if i != j:
                        edges.append((i, j))
                        weights.append(1 - d)   # cosine distance → similarity

            g = ig.Graph(edges=edges, directed=False)
            g.es["weight"] = weights

            # ---------------------------------------------------------
            # 3. Run Leiden clustering
            # ---------------------------------------------------------

            partition = la.find_partition(
                g,
                la.RBConfigurationVertexPartition,
                weights=g.es["weight"],
                resolution_parameter=1.0   # tune this for more/fewer clusters
            )
            # generated cluster labels via leiden clustering
            labels = np.array(partition.membership)

            # running UMAP reduction with cosine metric
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42, metric='cosine')

        elif clustering == 'kmeans':
            if k is None:
                # then do kmeans clustering if not leiden clustering 
                kmeans = KMeans(n_clusters=C_num.shape[1], random_state=42)
                labels = kmeans.fit_predict(C_num)
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(C_num)
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
        else: 
            raise ValueError("Clustering method not recognized. Use 'leiden' or 'kmeans'.")
        
        print(f"Clustering completed using {clustering} method.")
        # Umap reduction 
        
        F_umap = reducer.fit_transform(C_num)  # shape (p_features, 2)

            # Build DataFrame for Plotly
        df = pd.DataFrame({
                "UMAP1": F_umap[:, 0],
                "UMAP2": F_umap[:, 1],
                "Cluster": labels.astype(str),   # convert to string for discrete colors
                "Feature": self.feature_names
            })

        # Create interactive scatter plot
        fig = px.scatter(
                df,
                x="UMAP1", y="UMAP2",
                color="Cluster",
                hover_name="Feature",   # shows feature name on hover
                title="Leiden + Normalized feature clustering",
                labels={"UMAP1": "UMAP Dimension 1", "UMAP2": "UMAP Dimension 2"}
            )

        # Optional: tweak marker size and transparency
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0)))

        # Show interactive plot
        #fig.write_html('Kmeans_no_Normalization.html')
        if plot:
            fig.show()

        if save_file: 
            fig.write(f'{clustering}.html')
        # mean loadings per label...first get cluster df 
        cov_cluster_df = pd.DataFrame(
            C_num, 
            columns=[f'Spac-{i+1}' for i in range(C_num.shape[1])], 
            index= self.feature_names
            )
        cov_cluster_df['labels'] = labels
        
        # then get the actual average loading vectors 
        avg_loadings = cov_cluster_df.groupby('labels').mean()

        # matrix multiplication and then plot 
        proj_Avgloadings = self._cache['Vk'] @ avg_loadings.values.T

        # plot the projected average loadings
        self.plot_spatial_heatmap(
            proj_Avgloadings[:,0], 
            point_size=10, 
            cmap='Spectral',
            title="Avg_loading_projection_1"
            ) 

        return

    def plot_spatial_heatmap(
        self,
        values,
        marker = 'o',
        title="Spatial Heatmap",
        cmap="viridis",
        point_size=50,
        rotate_coords=False,
    ):
        """
        Plots a discrete spatial heatmap using scatter plot visualization.

        This function visualizes spatial data by plotting points at specified coordinates,
        colored according to their associated values. It is useful for displaying spatial
        patterns, such as gene expression, sensor measurements, or any data with spatial context.

        Args:
            values (numpy.ndarray):
                Array of shape (n_samples,) containing the values to be visualized at each coordinate.
            title (str, optional):
                Title of the plot. Defaults to "Spatial Heatmap".
            cmap (str, optional):
                Name of the matplotlib colormap to use for coloring the points. Defaults to "viridis".
            point_size (int, optional):
                Size of the points in the scatter plot. Defaults to 50.

        Returns:
            None. Displays the heatmap plot.

        Example:
            >>> coords = np.array([[0, 0], [1, 1], [2, 2]])
            >>> values = np.array([0.1, 0.5, 0.9])
            >>> plot_spatial_heatmap(coords, values, title="Example Heatmap", point_size=30)

        Notes:
            - The function assumes that `coords` and `values` are NumPy arrays of compatible shapes.
            - The color of each point reflects the corresponding value in the `values` array.
            - Useful for visualizing spatial patterns in 2D
        """
        # if user decides to rotate the coordinates, we do so
        if rotate_coords:
            self.coords = self.__rotate_coordinates(self.coords)

        # Ensure values is a numpy array or pandas DataFrame
        values = np.array(values)

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            c=values,
            cmap=cmap,
            s=point_size,
            marker = marker

        )
        plt.colorbar(scatter, label="Values")
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.show()

    def __rotate_coordinates(self, coords) -> np.ndarray:
        """
        Rotate the coordinates of the SpaCo object by 90 degrees counterclockwise.

        This method modifies the `coords` attribute of the SpaCo object in place.
        It does not return a value.

        Notes:
            - The rotation is performed by multiplying the coordinates with a rotation matrix.
            - The rotation matrix is defined as [[0, 1], [-1, 0]], which corresponds to a 90-degree counterclockwise rotation.
        """
        rotation_matrix = np.array([[0, 1], [-1, 0]])
        # Ensure coords is a numpy array or pandas DataFrame
        if isinstance(coords, pd.DataFrame):
            coords = coords.to_numpy()

        # Ensure inputs are NumPy arrays and coords is 2D with shape (n_samples, 2)
        coords = coords @ rotation_matrix.T
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must be a 2D array with shape (n_samples, 2)")
        return coords

def generate_sample_data(n_samples=80, n_features=100, seed=0, k_neighbors=10):
    """
    Returns:
      X: (n_samples, n_features) sample feature matrix (float)
      A: (n_samples, n_samples) symmetric binary adjacency matrix (int)
      coords: (n_samples, 2) random 2D coordinates (float)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # random 2D coordinates in [0,1)^2
    coords = rng.random((n_samples, 2))

    # pairwise distances
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    # k-nearest neighbor binary adjacency (no self-loops), symmetrized
    A = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        knn = np.argsort(dists[i])[1 : (1 + k_neighbors)]
        A[i, knn] = 1
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    return X, A, coords

if __name__ == "__main__":
    from sklearn.cluster import KMeans
    # generating fake data
    X, A, coords = generate_sample_data()

    spaco = SPACO(X, A, coords)

    # call feature extraction method 
    _ = spaco.spaco_projection()

    # time to inspect the features
    # something to notice is that we call spaco_test for each feature in 
    spaco.features_in_spacs(n_neighbors=5, clustering='leiden', normalization=True, plot=True, save_file=False)