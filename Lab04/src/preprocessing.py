"""Train-fitted preprocessing for the Adult classification pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class AdultPreprocessor(BaseEstimator, TransformerMixin):
    """Impute, train-quantile clip, encode, and scale Adult features."""

    def __init__(
        self,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        engineer_features: bool = False,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.engineer_features = engineer_features

    def _with_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if self.engineer_features:
            data["capital_net"] = data["capital-gain"] - data["capital-loss"]
            denominator = data["age"] - 16
            data["hours_per_age"] = data["hours-per-week"].div(
                denominator.where(denominator > 0)
            )
        return data

    def _numeric_array(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.numeric_features].to_numpy(dtype=float)
        values = np.clip(values, self.lower_bounds_, self.upper_bounds_)
        return np.where(np.isnan(values), self.numeric_medians_, values)

    def _categorical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.loc[:, self.categorical_features].astype(object).copy()
        data = data.replace("?", np.nan)
        for column, mode in self.categorical_modes_.items():
            data[column] = data[column].where(data[column].notna(), mode)
        return data.astype(str)

    def fit(self, X: pd.DataFrame, y: Any = None) -> AdultPreprocessor:
        data = self._with_features(X)
        numeric = data.loc[:, self.numeric_features].to_numpy(dtype=float)
        self.lower_bounds_ = np.nanpercentile(numeric, 1, axis=0)
        self.upper_bounds_ = np.nanpercentile(numeric, 99, axis=0)
        clipped = np.clip(numeric, self.lower_bounds_, self.upper_bounds_)
        self.numeric_medians_ = np.nanmedian(clipped, axis=0)
        if np.isnan(self.numeric_medians_).any():
            raise ValueError("A numeric feature is entirely missing in training data.")

        categorical = data.loc[:, self.categorical_features].astype(object)
        categorical = categorical.replace("?", np.nan)
        self.categorical_modes_ = {}
        for column in self.categorical_features:
            mode = categorical[column].mode(dropna=True)
            if mode.empty:
                raise ValueError(
                    f"Categorical feature {column!r} is entirely missing in training data."
                )
            self.categorical_modes_[column] = mode.iloc[0]

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.encoder_.fit(self._categorical_frame(data))
        numeric_train = self._numeric_array(data)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(numeric_train)
        self.feature_names_out_ = np.concatenate(
            [
                np.asarray(self.numeric_features, dtype=object),
                self.encoder_.get_feature_names_out(self.categorical_features),
            ]
        )
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        check_is_fitted(self, ["lower_bounds_", "encoder_", "scaler_"])
        data = self._with_features(X)
        numeric = self.scaler_.transform(self._numeric_array(data))
        categorical = self.encoder_.transform(self._categorical_frame(data))
        return sparse.hstack(
            [sparse.csr_matrix(numeric), categorical], format="csr"
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
