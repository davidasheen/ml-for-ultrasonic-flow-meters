import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy

import sklearn.ensemble as skens
import sklearn.svm as sksvm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.utils import estimate_dof
from pychemauth.classifier.simca import SIMCA_Authenticator
from pychemauth.classifier.osr import OpenSetClassifier


class SVM_auth_wrapper(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        target_class=None,
        **svm_kw
    ):
        self.target_class = target_class
        self.model = sksvm.SVC(**svm_kw)
        
    def fit(self,X,y,):
        self.model.fit(X,y)
        return self
    def predict(self,X):
        y_pred = self.model.predict(X)
        return y_pred == self.target_class

class SVC_Auth(SIMCA_Authenticator):
    def __init__(
        self,
        C=1,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        target_class=None,
        use="rigorous",
        scale_x=True,
        robust="semi",
        sft=False,
    ):
        self.set_params(
            **{
                "C": C,
                "kernel": kernel,
                "degree": degree,
                "gamma": gamma,
                "coef0": coef0,
                "target_class": target_class,
                "use": use,
                "scale_x": scale_x,
                "robust": robust,
                "sft": sft,
            }
        )
        
    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "target_class": self.target_class,
            "use": self.use,
            "scale_x": self.scale_x,
            "robust": self.robust,
            "sft": self.sft,
        }
    def fit(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
            copy=False,
        )
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([True, False])  # For sklearn compatibility
        
        #self.__model_ = sklearn.svm.SVC(
        self.__model_ = SVM_auth_wrapper(
            target_class=self.target_class,
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
        )
        
        self.__model_.fit(X,y)
        self.is_fitted_ = True
        self._SIMCA_Authenticator__model_ = self.__model_ 
        
        #return self.__model_
        
        return self
    
    def decision_function(self,X):
        
        decision_value = np.where(self.model.model.classes_==self.target_class)[0]
        
        full_d = self.model.model.decision_function(X)
        scaled_d = full_d[:,decision_value]
        return scaled_d
    
    def score(self,X,y):
        check_is_fitted(self, "is_fitted_")
        m = self.metrics(X, y)
        return m["TEFF"]
        
class RF_auth_wrapper(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        target_class=None,
        **rf_kw
    ):
        self.target_class = target_class
        self.model = skens.RandomForestClassifier(**rf_kw)
        
    def fit(self,X,y,):
        self.model.fit(X,y)
        return self
    def predict(self,X):
        y_pred = self.model.predict(X)
        return y_pred == self.target_class

class RF_Auth(SIMCA_Authenticator):
    def __init__(
        self,
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='sqrt',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        target_class=None,
        use="rigorous",
        scale_x=True,
        robust="semi",
        sft=False,
    ):
        self.set_params(
            **dict(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                target_class=target_class,
                use=use,
                scale_x=scale_x,
                robust=robust,
                sft=sft,
            )
        )
        
    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return dict(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            target_class=self.target_class,
            use=self.use,
            scale_x=self.scale_x,
            robust=self.robust,
            sft=self.sft,
        )
    def fit(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
            copy=False,
        )
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([True, False])  # For sklearn compatibility
        
        #self.__model_ = sklearn.svm.SVC(
        self.__model_ = RF_auth_wrapper(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            target_class=self.target_class,
        )
        
        self.__model_.fit(X,y)
        self.is_fitted_ = True
        self._SIMCA_Authenticator__model_ = self.__model_ 
        
        #return self.__model_
        
        return self
    
    def score(self,X,y):
        check_is_fitted(self, "is_fitted_")
        m = self.metrics(X, y)
        return m["TEFF"]

class OSC_auth_wrapper(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        target_class=None,
        **svm_kw
    ):
        self.target_class = target_class
        self.model = OpenSetClassifier(**svm_kw)
        
    def fit(self,X,y,):
        self.model.fit(X,y)
        return self
    def predict(self,X):
        y_pred = np.array(self.model.predict(X))
        return y_pred == self.target_class

class OSC_Auth(SIMCA_Authenticator):
    def __init__(
        self,
        clf_model=None,
        outlier_model=None,
        clf_kwargs={},
        outlier_kwargs={},
        known_classes=None,
        inlier_value=1,
        unknown_class="Unknown",
        score_metric="TEFF",
        clf_style="hard",
        score_using="all",
        target_class=None,
    ):
        self.set_params(
            **{
                "clf_model": clf_model,
                "outlier_model": outlier_model,
                "clf_kwargs": clf_kwargs,
                "outlier_kwargs": outlier_kwargs,
                "known_classes": known_classes,
                "inlier_value": inlier_value,
                "unknown_class": unknown_class,
                "score_metric": score_metric,
                "clf_style": clf_style,
                "score_using": score_using,
                "target_class": target_class,
            }
        )
        
    def get_params(self, deep=True):
        """Get parameters; for consistency with scikit-learn's estimator API."""
        return {
            "clf_model": self.clf_model,
            "outlier_model": self.outlier_model,
            "clf_kwargs": self.clf_kwargs,
            "outlier_kwargs": self.outlier_kwargs,
            "known_classes": self.known_classes,
            "inlier_value": self.inlier_value,
            "unknown_class": self.unknown_class,
            "score_metric": self.score_metric,
            "clf_style": self.clf_style,
            "score_using": self.score_using,
            "target_class": self.target_class,
        }
    def fit(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            y_numeric=False,
            copy=False,
        )
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([True, False])  # For sklearn compatibility
        
        #self.__model_ = sklearn.svm.SVC(
        self.__model_ = OSC_auth_wrapper(
            target_class=self.target_class,
            clf_model=self.clf_model,
            outlier_model=self.outlier_model,
            clf_kwargs=self.clf_kwargs,
            outlier_kwargs=self.outlier_kwargs,
            known_classes=self.known_classes,
            inlier_value=self.inlier_value,
            unknown_class=self.unknown_class,
            score_metric=self.score_metric,
            clf_style=self.clf_style,
            score_using=self.score_using,
        )
        
        self.__model_.fit(X,y)
        self.is_fitted_ = True
        self._SIMCA_Authenticator__model_ = self.__model_ 
        
        #return self.__model_
        
        return self
    
    def score(self,X,y):
        check_is_fitted(self, "is_fitted_")
        m = self.metrics(X, y)
        return m["TEFF"]

from sklearn.inspection import DecisionBoundaryDisplay

def plot_decision_info(
    ax,model,transformer,X,y,target_class,label='',
    positive_color='g',negative_color='b',
    true_color='g',false_color='r',
):
    y_pred = model.predict(X)
    
    y_class = y==target_class
    
#     #Color the points in X by their class
#     colors_class = np.array(['b']*y.shape[0])
#     colors_class[y_class] = 'g'
#     colors_class
    
#     #Color the points in X by whether they are correctly classified
#     colors_valid = np.array(['r']*y.shape[0])
#     colors_valid[y_pred==y_class] = 'g'
#     colors_valid
    
    scores = transformer.transform(X)
    
    ax.plot(
        scores[y_pred&y_class,0],scores[y_pred&y_class,1],
        marker='o',
        markerfacecolor=positive_color,
        markeredgecolor=true_color,
        ls='none',
        label='True positive'+label
    )

    ax.plot(
        scores[y_pred&~y_class,0],scores[y_pred&~y_class,1],
        marker='o',
        markerfacecolor=positive_color,
        markeredgecolor=false_color,
        ls='none',
        label='False negative'+label
    )

    ax.plot(
        scores[~y_pred&y_class,0],scores[~y_pred&y_class,1],
        marker='o',
        markerfacecolor=negative_color,
        markeredgecolor=false_color,
        ls='none',
        label='False positive'+label
    )

    ax.plot(
        scores[~y_pred&~y_class,0],scores[~y_pred&~y_class,1],
        marker='o',
        markerfacecolor=negative_color,
        markeredgecolor=true_color,
        ls='none',
        label='True negative'+label
    )
    
    return scores,y_pred

def svc_acceptance_plot(ax,clf,data,y,transformer,classes):
    
    markers = [
        "o",
        "v",
        "s",
        "*",
        "+",
        "x",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        "4",
    ]
    
    for this_class,marker in zip(classes,markers):
        is_this_class = y==this_class
        scores = transformer.transform(data[is_this_class])
        y_pred = clf.predict(scores)
        if np.any(y_pred):
            ax.plot(
                scores[y_pred,0],scores[y_pred,1],
                label='{} = inlier ({})'.format(this_class,np.count_nonzero(y_pred)),
                marker=marker,ls='none',c="g",alpha=0.35
            )
        if np.any(~y_pred):
            ax.plot(
                scores[~y_pred,0],scores[~y_pred,1],
                label='{} = outlier ({})'.format(this_class,np.count_nonzero(~y_pred)),
                marker=marker,ls='none',c="r",alpha=0.35
            )
    
    scores = transformer.transform(data)
    
    common_params = {
        "estimator": clf,
        #"estimator": gs_scaled_svc.best_estimator_['svc'].model.model,
        "X": scores,
        "ax": ax
    }
    
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["g", "g", "g"],
        linestyles=["--", "-", "--"],#label='Decision boundary'
        xlabel='PC 1',ylabel='PC 2'
    )
    
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="0.65",label='Support vectors'
    )
    ax.legend()