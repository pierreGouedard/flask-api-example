# Global imports
import itertools
import logging
import pandas as pd
import numpy as np
import pickle as pickle
import os
from sklearn.metrics import precision_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Iterator, Any, Optional, Union

# Local import
from .names import KVName
from .features import FoldManager


class ClfSelector(object):
    """
    The ClfSelector is an object that supervise the selection of hyper parameter of classifier using cross val.

    """
    allowed_score = ['precision', 'accuracy', 'balanced_accuracy']

    def __init__(
            self, df_data: pd.DataFrame, param_mdl: Dict[str, Any], param_mdl_grid: Dict[str, Any],
            param_transform: Dict[str, Any], param_transform_grid: Dict[str, Any], params_fold: Dict[str, Any],
            scoring: str

    ) -> None:
        """

        Parameters
        ----------
        df_data : pandas.DataFrame
            DataFrame composed of document text and label.

        param_transform : dict
            Dictionary of fixed parameter for transforming input data.

        param_transform_grid : dict
            Dictionary of parameter for transforming input data to select using a grid search.

        param_mdl : dict
            Dictionary of fixed parameter for model.

        param_mdl_grid : dict
            Dictionary of parameter for model to select using a grid search.

        params_fold : dict
            Dictionary of parameter to use to make the cross validation of features builder and model.

        scoring : str
            Scoring method to use to evaluate models and models builder.

        """
        assert scoring in self.allowed_score, "Choose scoring among {}".format(self.allowed_score)

        # Core parameter classifier
        self.scoring = scoring

        # Parameters of models builder and model
        self.param_transform, self.param_transform_grid = param_transform, param_transform_grid
        self.param_mdl, self.param_mdl_grid = param_mdl, param_mdl_grid

        # Core attribute of classifier
        self.fold_manager = FoldManager(df_data, **params_fold)
        self.is_fitted, self.model, self.d_search = False, None, None

    def fit(self) -> Iterator[Union[Dict[str, Any], bool]]:
        """
        Find the optimal combination of hyper parameter of models building and model through grid search. Fit the
        entire etl with the selected parameters.

        """
        # Fit model and models builder using K-fold cross validation
        for d_search in self.grid_search():
            yield d_search

        self.d_search = d_search
        self.model = self.__fit_all()
        self.is_fitted = True

        yield True

    def grid_search(self) -> Iterator[Dict[str, Any]]:
        """
        Perform a cross validation of hyper parameters of the features building routine and the model used for
        classification. For given parameter of models builder, we retain the parameter of prediction model that get
        the best score. The parameter of models builder with the higher score will be chosen, along with the best model
        associated.

        """

        if len(self.param_transform_grid) <= 1 and len(self.param_mdl_grid) <= 1:
            yield {0: {"params_feature": self.param_transform, "param_mdl": self.param_mdl}, 'best_key': 0}

        d_search, best_score = {}, 0.
        for i, cross_values in enumerate(itertools.product(*self.param_transform_grid.values())):

            # Reset fold_manager to allow fit of new models builder
            self.fold_manager.reset()

            # Update params of grid search
            d_features_grid_instance = self.param_transform.copy()
            d_features_grid_instance.update(dict(zip(self.param_transform_grid.keys(), cross_values)))
            d_search[i] = {'params_feature': d_features_grid_instance.copy()}

            # Inform about Current params
            logging.info("[FEATURE]: {}".format(KVName.from_dict(d_features_grid_instance).to_string()))

            # Fit model and keep params of best model associated with current models's parameters
            best_mdl_params, best_mdl_score = self.grid_search_mdl(d_features_grid_instance)
            d_search[i].update({'param_mdl': best_mdl_params, 'best_score': best_mdl_score})

            yield d_search

            # Keep track of best association models builder / model
            if best_score < best_mdl_score:
                d_search['best_key'] = i

        logging.info('Optimal parameters found are {}'.format(d_search[d_search['best_key']]))

    def grid_search_mdl(self, d_feature_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a cross validation of hyper parameter of the classifier

        Parameters
        ----------
        d_feature_params : dict
            Dictionary of parameters for building features.

        Returns
        -------
        tuple
            A tuple containing a dictionary of parameters of the best model and the score of the best model
        """
        best_mdl_params, best_mdl_score = None, 0.
        for cross_values in itertools.product(*self.param_mdl_grid.values()):

            # Update params of logistic regression model
            d_mdl_grid_instance = self.param_mdl.copy()
            d_mdl_grid_instance.update(dict(zip(self.param_mdl_grid.keys(), cross_values)))

            # Instantiate model and cross validate the hyper parameter
            model, l_errors = RandomForestClassifier(**d_mdl_grid_instance), []
            for d_train, d_val in self.fold_manager.generate_folds(d_feature_params):

                # Fit model and save score
                model.fit(d_train['X'], d_train['y'])
                l_errors.append(get_score(self.scoring, model.predict(d_val['X']), d_val['y']))

            # Average errors and update best params if necessary
            mu = np.mean(l_errors)
            logging.info('[MODEL]: {} | score: {}'.format(KVName.from_dict(d_mdl_grid_instance).to_string(), mu))

            if best_mdl_score < mu:
                best_mdl_params, best_mdl_score = d_mdl_grid_instance.copy(), mu

        return best_mdl_params, best_mdl_score

    def __fit_all(self) -> object:
        """
        Fit models builder and model based on optimal hyper parameter found.

        :return: trained model

        """
        # Reset data_manager to allow fit of we
        self.fold_manager.reset()

        # Recover optimal parameters
        d_transform_param = self.d_search[self.d_search['best_key']]['params_feature']
        d_model_params = self.d_search[self.d_search['best_key']]['param_mdl']
        d_train = self.fold_manager.get_all_train_data(d_transform_param)

        # Instantiate model and fit it
        model = RandomForestClassifier(**d_model_params)\
            .fit(d_train['X'], d_train['y'])

        return model

    def get_classifier(self) -> "Classifier":
        """
        Return classifier.
        :return: Classifier
        """
        # Get best params for feature and model
        d_param_feature = self.d_search[self.d_search['best_key']]['params_feature']
        d_param_model = self.d_search[self.d_search['best_key']]['param_mdl']

        return Classifier(self.model, self.fold_manager.data_transformer, d_param_model, d_param_feature)

    def save_classifier(self, path: str) -> "ClfSelector":
        """
        Save core element of the documetn classifier.

        Parameters
        ----------
        path : str
            path toward the location where classifier shall be saved.

        Returns
        -------

        """
        # Get and pickle classifier
        with open(path, 'wb') as handle:
            pickle.dump(self.get_classifier(), handle)

        return self

    def save_data(self, path: str, name_train: str, name_test: str):
        """
        Save data used to select and fit the classifier.

        Parameters
        ----------
        path : str
            path toward the location where classifier shall be saved.
        name_train : str
            name of file containing train data.
        name_test : str
            name of file containing test data.

        Returns
        -------

        """

        path_train, path_test = os.path.join(path, name_train), os.path.join(path, name_test)
        self.fold_manager.df_train.to_hdf(path_train, key=name_train.split('.')[0], mode='w')
        self.fold_manager.df_test.to_hdf(path_test, key=name_test.split('.')[0], mode='w')

        return self


class Classifier(object):
    """
    The Classifier is an object that ready to use to classify.

    """

    def __init__(
            self, model: object, data_transformer: object, param_model: Dict[str, Any], param_transform: Dict[str, Any]
    ) -> None:
        """

        Parameters
        ----------
        model : object
            Fitted model to use to classify documents.

        data_transformer : src.model.feature.FeatureBuilder
            Fitted model to use to vectorize text of documents.

        param_transform : dict
            Dictionary of fixed parameter for building features.

        param_model : dict
            Dictionary of fixed parameter for classification model.


        """

        # Core parameter classifier
        self.model = model
        self.data_transformer = data_transformer
        self.param_model = param_model
        self.param_transform = param_transform

    @staticmethod
    def from_path(path: str) -> "Classifier":
        """
        Load Classifier from pickle file.

        :param path: str
        :return:
        """
        with open(path, 'rb') as handle:
            dc = pickle.load(handle)

        return Classifier(**dc.__dict__)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict target space for feature in df.

        Parameters
        ----------
        df : DataFrame
            Dataframe containing features

        Returns
        -------
        Series
           Series with predicted target values.
        """
        features = self.data_transformer.transform(df)
        preds = self.model.predict(features)

        return pd.Series(self.data_transformer.target_encoder.inverse_transform(preds), index=df.index, name="preds")


def get_score(
        scoring: str, yhat: np.array, y: np.array, average: str = 'macro', labels: Optional[List[str]] = None
) -> float:
    """
    Compute classification score from ground of truth and prediction.

    :param scoring: str, name of the scoring metric
    :param yhat: array, contains predicted target.
    :param y: array, contains ground of truth.
    :param average: str, method to use for multi target classification
    :param labels: list, list of label that can take the target.
    :return: float, score.

    """
    if scoring == 'precision':
        if labels is None and average is not None:
            labels = list(set(y).intersection(set(yhat)))

        score = precision_score(y, yhat, labels=labels, average=average)

    elif scoring == 'accuracy':
        score = accuracy_score(y, yhat)

    elif scoring == 'balanced_accuracy':
        score = balanced_accuracy_score(y, yhat)

    else:
        raise ValueError('Scoring name not understood: {}'.format(scoring))

    return score
