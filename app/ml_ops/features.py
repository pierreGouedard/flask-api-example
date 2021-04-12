# Global imports
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Iterator

# Local import


class FoldManager(object):
    """
    FoldManager manage segmentation of data for cross validation.

    """
    allowed_methods = ['standard', 'stratified']

    def __init__(
            self, df_data: pd.DataFrame, nb_folds: int, test_size: float = 0.1, target_name: str = 'target'
    ) -> None:

        # Get base parameters
        self.target_name = target_name

        # Split data set into a train / test and Validation if necessary
        self.df_train, self.df_test = train_test_split(df_data, test_size=test_size, shuffle=True)

        # Set method to transform data
        self.data_transformer = None

        # Set sampling method for Kfold
        self.kf = KFold(n_splits=max(nb_folds, 3), shuffle=True)

    def reset(self):
        """
        Reset fold manager by setting data transformer to None.
        """
        self.data_transformer = None

    def get_all_train_data(self, param_transform: Dict[str, Any], force_recompute: bool = False) -> Dict[str, np.array]:
        """
        Build a data set composed of models. The target is also return, if specified.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        force_recompute : bool
            If True, it fit feature builder with train data

        Returns
        -------
        dict

        """
        # concat train and validation if any
        df_train = self.df_train

        # Create models builder if necessary
        if self.data_transformer is None or force_recompute:
            self.data_transformer = DataTransformer(**param_transform).build(df_train)

        X, y = self.data_transformer.transform(df_train, target=True)

        return {"X": X, "y": y}

    def get_test_data(self, param_transform: bool = None) -> Dict[str, np.array]:
        """
        Build test data set composed of models and transformed target.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        Returns
        -------
        dict
            Features and target as

        """
        # Create models builder if necessary
        if self.data_transformer is None:
            self.data_transformer = DataTransformer(**param_transform).build(self.df_train)

        X, y = self.data_transformer.transform(self.df_test, target=True)

        return {"X": X, "y": y}

    def generate_folds(self, param_transform: Dict[str, Any]) -> Iterator[Dict[str, np.array]]:
        """
        Generate train and validation data set. A data set is composed of models and target.

        Parameters
        ----------
        param_transform : dict
            kw parameters to build features.

        Returns
        -------
        tuple
            Composed of dict with Features and target, for train and validation etl.
            i.e: {'X': numpy.ndarray, 'y': numpy.ndarray}

        """
        # Iterate over different folds
        for l_train, l_val in self.kf.split(self.df_train):
            index_train, index_val = list(self.df_train.index[l_train]), list(self.df_train.index[l_val])

            # Create models  builder if necessary
            self.data_transformer = DataTransformer(**param_transform).build(self.df_train.loc[index_train])

            # Get features
            X, y = self.data_transformer.transform(self.df_train.loc[index_train + index_val], target=True)

            # Build train / validation set
            X_train, y_train = X[l_train, :], y[l_train]
            X_val, y_val = X[l_val, :], y[l_val]

            yield {'X': X_train, 'y': y_train}, {'X': X_val, 'y': y_val}


class DataTransformer(object):
    """
    The DataTransformer manage the transformation of processed data composed of job description labelled by normalized
    positions. Its transformation pipeline is composed of:
    """
    def __init__(
            self, feature_transform: str, cat_cols: List[str], num_cols: List[str], target_col: str,
            params: Dict[str, Any], labels: List[str], target_transform: str = None
    ) -> None:

        self.labels, self.target_col, self.cat_cols, self.num_cols = labels, target_col, cat_cols, num_cols
        self.feature_transform, self.target_transform = feature_transform, target_transform
        self.params, self.args = params, {}

        # Init model to None
        self.model, self.target_encoder, self.imputer = None, None, None

    def clean(self, df: pd.DataFrame, check_target: bool = False) -> pd.DataFrame:
        """
        Clean passed DataFrame.

        :param df: DataFrame, Data to clean.
        :param check_target: bool, specify if target need to be cleaned.
        :return: DataFrame, clean DataFrame.

        """
        # Clean numeric features
        df = df.assign(
            **{c: lambda x: np.where(x[c].str.match('^[0-9]+.[0-9]*$'), x[c], np.nan) for c in self.num_cols}
        ).astype({**{c: float for c in self.num_cols}})

        # Remove Nan target if necessary
        if check_target:
            df = df.assign(
                **{self.target_col: lambda x: np.where(x[self.target_col].isna(), 'unidentified', x[self.target_col])}
            )

        return df

    def build(self, df_data: pd.DataFrame = None, force_train: bool = False) -> "DataTransformer":
        """
        Create and fit models to preprocess data for training.

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame data to process and to fit the transformation model upon.
        params :  dict
            Kw params of the transformations
        force_train : bool
            whether to force the fit of transformation model.

        Returns
        -------
        self : Current instance of the class.
        """
        # Clean data
        df_data = self.clean(df_data, check_target=True)

        if self.model is None or force_train:

            if 'standardize' in self.feature_transform:
                pass

            if 'impute' in self.feature_transform:
                self.imputer = {}

                if len(self.cat_cols) > 0:
                    self.imputer['cat_imput'] = SimpleImputer(**self.params.get('impute_cat', {}))\
                        .fit(df_data[self.cat_cols])

                if len(self.num_cols) > 0:
                    self.imputer['num_imput'] = SimpleImputer(**self.params.get('impute_num', {}))\
                        .fit(df_data[self.num_cols])

            else:
                raise ValueError('Transformation not implemented: {}'.format(self.feature_transform))

            if self.target_transform == 'encode':
                self.target_encoder = LabelEncoder().fit(self.labels)

        return self

    def transform(self, df_data, target=False):
        """
        Use previously fit model to transform feature of passed DataFrame
        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame of that shall be transformed.

        target : bool
            specifies if target transform is necessary.

        Returns
        -------
        numpy.array
            Transformed data.

        """
        # Clean data
        df_data = self.clean(df_data, check_target=True)

        if 'impute' in self.feature_transform:
            if self.imputer.get("cat_imput", None) is not None:
                X_cat = self.imputer['cat_imput'].transform(df_data[self.cat_cols])
            else:
                X_cat = df_data[self.cat_cols]

            if self.imputer.get("num_imput", None) is not None:
                X_num = self.imputer['num_imput'].transform(df_data[self.num_cols])
            else:
                X_num = df_data[self.num_cols]

        else:
            X_cat, X_num = df_data[self.cat_cols], df_data[self.num_cols]

        X = np.hstack([X_cat, X_num])

        if target:
            if self.target_transform == 'encode':
                y = self.target_encoder.transform(df_data[self.target_col])

            else:
                y = df_data.loc[:, [self.target_col]].values

            return X, y

        return X
