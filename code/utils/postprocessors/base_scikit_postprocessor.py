
import torch
from typing import Dict, Type

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from .base_postprocessor import BasePostprocessor

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g



class BaseScikitPostprocessor(BasePostprocessor):
    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        """
        Args:
            classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
            weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
            means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            n_cluster (int): Number of clusters to partition the error probability into.
            alpha (float): Confidence level parameter for interval widths.
            method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
            seed (int): Random seed for data generation.
            device (torch.device): Device on which to run the classifier.
        """
        super().__init__(model, cfg, result_folder, device)

        self.quantiz_space = cfg["space"]
        self.reorder_embs = cfg["reorder_embs"]
        self.temperature = cfg["temperature"]
        # self.regressor_name = cfg["regressor_name"]

    @torch.no_grad()
    def _extract_embeddings(self, x=None, logits=None):
        """
        Extract embeddings from the model.
        This function is used to create a feature extractor.
        """

        # Should implement the reducer here!
        # if self.reducer is not None:
        #     all_embs = torch.tensor(self.reducer.fit_transform(all_embs.cpu().numpy()), device=self.device)

       
        if logits is not None:
            # if self.class_subset is not None:
            #     logits = logits[:, self.class_subset]
            if self.quantiz_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
            elif self.quantiz_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
            elif self.quantiz_space == "logits":
                embs = logits
        else:
            self.model.to(self.device)
            logits = self.model(x)
            # if self.class_subset is not None:
            #     logits = logits[:, self.class_subset]
            self.model.to(torch.device('cpu'))
            if self.quantiz_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
            elif self.quantiz_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
            elif self.quantiz_space == "logits":
                embs = logits
            else:
                raise ValueError("Unsupported quantiz_space")
            
         # Reorder embeddings if needed
        if self.reorder_embs:
            embs, idx = embs.sort(dim=1, descending=True)  # idx: shape (B, N)
            self._perm_idx = idx

        return embs

    # ---------- Fit / Inference ----------
    @torch.no_grad()
    def fit(self, logits=None, detector_labels=None, verbose=False):

    
        embs = self._extract_embeddings(logits=logits)
        self.regressor.fit(
            X= embs.detach().cpu().numpy(), 
            y=detector_labels.detach().cpu().numpy()
        )
        return self

    @torch.no_grad()
    def __call__(self, x=None, logits=None):
        """
        Returns P(error | embedding) as a NumPy array of shape [N].
        """
        if logits is None:
            if x is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(x)

        embs = self._extract_embeddings(x=x, logits=logits).detach().cpu().numpy()
        proba = self.regressor.predict_proba(embs)  # [:, 1] is class "1" (error)
        return torch.tensor(proba[:, 1]).to(self.device)
    


class GenericScikitPostprocessor(BaseScikitPostprocessor):
    """
    Lightweight wrapper that exposes a variety of scikit-learn classifiers as
    post-hoc misclassification detectors within the existing pipeline.

    The estimator is selected through `cfg["estimator"]` and instantiated with
    default hyper-parameters unless `cfg["estimator_kwargs"]` is provided.
    """

    _ESTIMATORS: Dict[str, Type] = {
        "dummy_classifier": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "logistic_regression_cv": LogisticRegressionCV,
        "k_neighbors_classifier": KNeighborsClassifier,
        "decision_tree_classifier": DecisionTreeClassifier,
        "extra_tree_classifier": ExtraTreeClassifier,
        "random_forest_classifier": RandomForestClassifier,
        "extra_trees_classifier": ExtraTreesClassifier,
        "gradient_boosting_classifier": GradientBoostingClassifier,
        "hist_gradient_boosting_classifier": HistGradientBoostingClassifier,
        "ada_boost_classifier": AdaBoostClassifier,
        "bagging_classifier": BaggingClassifier,
        "gaussian_nb": GaussianNB,
        "multinomial_nb": MultinomialNB,
        "bernoulli_nb": BernoulliNB,
        "complement_nb": ComplementNB,
        "linear_discriminant_analysis": LinearDiscriminantAnalysis,
        "quadratic_discriminant_analysis": QuadraticDiscriminantAnalysis,
        "mlp_classifier": MLPClassifier,
        "calibrated_classifier_cv": CalibratedClassifierCV,
    }

    def __init__(self, model, cfg, result_folder, device=torch.device("cpu")):
        super().__init__(model, cfg, result_folder, device)

        estimator_key = cfg.get("estimator")
        if estimator_key is None:
            raise KeyError("`estimator` must be provided in the postprocessor config.")

        estimator_key = estimator_key.lower()
        if estimator_key not in self._ESTIMATORS:
            raise ValueError(
                f"Unknown scikit-learn estimator '{estimator_key}'. "
                f"Available keys: {sorted(self._ESTIMATORS)}"
            )

        estimator_cls = self._ESTIMATORS[estimator_key]

        base_keys = {
            "space",
            "temperature",
            "reorder_embs",
            "normalize_gini",
            "estimator",
            "estimator_kwargs",
        }
        extra_kwargs = {
            key: value for key, value in cfg.items()
            if key not in base_keys
        }
        estimator_kwargs = dict(cfg.get("estimator_kwargs", {}) or {})
        estimator_kwargs.update(extra_kwargs)
        if not isinstance(estimator_kwargs, dict):
            raise TypeError("`estimator_kwargs` must be a mapping if provided.")

        self.regressor = estimator_cls(**estimator_kwargs)
