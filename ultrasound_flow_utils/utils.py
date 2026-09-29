import numpy as np
import sklearn
import imblearn
import xgboost

import sklearn.decomposition as skdecomp
import sklearn.ensemble as skens
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.cross_decomposition as skpls
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold,StratifiedKFold,KFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.feature_selection import CollinearFeatureSelector
from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.eda.explore import InspectData

score_metrics = dict(
    mae=sklearn.metrics.mean_absolute_error,
    rmse=sklearn.metrics.root_mean_squared_error,
    #r2=robust_r2,
    r2=sklearn.metrics.r2_score,
    mape=sklearn.metrics.mean_absolute_percentage_error,
    medae=sklearn.metrics.median_absolute_error,
    maxe=sklearn.metrics.max_error,
)

def logabs(x):
    return np.log1p(np.abs(x))

def generate_models(xdata,thresh=0.0,log_scale=False,selection=False,selector=None,n_jobs=8):
    #common parameters for gridsearch

    gridsearch_kfold = KFold(n_splits=5,shuffle=True,random_state=99)

    gs_common_params = dict(
        cv = gridsearch_kfold,
        verbose=1,
        n_jobs=n_jobs,
        #scoring='r2',
        scoring='neg_mean_squared_error',
    )
    
    ols_params = {
        #'ols__tol': [1e-5,1e-6,1e-7]
        'ols__fit_intercept':[True,False],
        'ols__positive':[True,False],
    }

    #gridsearch parameter grid for PLS
    pls_n_components = min(20,len(xdata.columns.values)) # make sure number of PLS components is less than
    pls_params = {
        'pls__n_components': np.arange(1, pls_n_components, 1) # Search from 1 to 19 components
    }

    #gridsearch parameter grids for SVM
    c_dict = {
        'svm__C':np.logspace(-2,2,9) # regularization parameter
    }

    degree_dict ={
        'svm__degree':np.arange(2,4,1) # degree for polynomial svr
    }

    gamma_dict = {
        'svm__gamma':['scale','auto'] #
    }

    linear_svm_params = dict(**c_dict)
    poly_svm_params = dict(**c_dict,**degree_dict,)#**gamma_dict)
    rbf_svm_params = dict(**c_dict,**gamma_dict)

    #gridsearch parameter grids for linear models
    alpha_dict = {
        'est__alpha':np.logspace(-2,2,9) # regularization parameter (equivalent to 1/2C for SVM)
    }

    l1_ratio_dict = {
        'est__l1_ratio':1/(
            1+np.exp(-np.linspace(-4,4,9)) # l1 ratio for elastinet
        )
    }

    lasso_params = dict(**alpha_dict)
    ridge_params = dict(**alpha_dict,)
    elastic_params = dict(**alpha_dict,**l1_ratio_dict)

    forest_est_params = {
        'forest__n_estimators':np.arange(10,200,10),
        'forest__max_depth':list(np.arange(2,10,2))+[None],
        'forest__max_features':["sqrt", "log2", None],
    }
    
    xgboost_est_params = {
        'forest__num_parallel_tree':np.arange(1,20,4),
        'forest__max_depth':np.arange(2,10,4),
        'forest__learning_rate':np.logspace(-2,-1,3),
        'forest__subsample':np.linspace(0.6,1.0,3),
        'forest__reg_alpha':np.logspace(-1,1,5),
        'forest__reg_lambda':np.logspace(-1,1,5)
    }

    #model definitions
    #var_step = [('variance',sklearn.feature_selection.VarianceThreshold(threshold=thresh))]
    #var_step = [(
    #    'var',CoefficientofVariationThreshold(threshold=thresh)
    #)]
    var_step = []
    log_step = [(
        'log',sklearn.preprocessing.FunctionTransformer(logabs)
    )] if log_scale else []
    sel_step = [('selection',selector)] if selection else []
    pre_steps = var_step + sel_step + log_step
    
    #oridnary least squares
    ols=imblearn.pipeline.Pipeline(
        steps= pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("ols", sklin.LinearRegression(),)
        ]
    )
    
    ols_gs = GridSearchCV(
        estimator = ols,
        param_grid = ols_params,
        **gs_common_params
    )
    
    #PLS
    unscaled_pls=imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("pls", skpls.PLSRegression(),)
        ]
    )
    
    unscaled_pls_gs = GridSearchCV(
        estimator = unscaled_pls,
        param_grid = pls_params,
        **gs_common_params
    )
    
    #SVM
    linear_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("svm", sksvm.SVR(kernel='linear'),)
        ]
    )

    poly_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(with_std=False)),
            ("svm", sksvm.SVR(kernel='poly'),)
        ]
    )

    pareto_poly_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=True)),
            ("svm", sksvm.SVR(kernel='poly'),)
        ]
    )

    scaled_poly_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("svm", sksvm.SVR(kernel='poly'),)
        ]
    )

    rbf_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(with_std=False)),
            ("svm", sksvm.SVR(kernel='rbf'),)
        ]
    )

    pareto_rbf_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=True)),
            ("svm", sksvm.SVR(kernel='rbf'),)
        ]
    )

    scaled_rbf_svm = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("svm", sksvm.SVR(kernel='rbf'),)
        ]
    )

    linear_svm_gs = GridSearchCV(
        estimator = linear_svm,
        param_grid = linear_svm_params,
        **gs_common_params
    )

    poly_svm_gs = GridSearchCV(
        estimator = poly_svm,
        param_grid = poly_svm_params,
        **gs_common_params
    )

    pareto_poly_svm_gs = GridSearchCV(
        estimator = pareto_poly_svm,
        param_grid = poly_svm_params,
        **gs_common_params
    )

    scaled_poly_svm_gs = GridSearchCV(
        estimator = scaled_poly_svm,
        param_grid = poly_svm_params,
        **gs_common_params
    )

    rbf_svm_gs = GridSearchCV(
        estimator = rbf_svm,
        param_grid = rbf_svm_params,
        **gs_common_params
    )

    pareto_rbf_svm_gs = GridSearchCV(
        estimator = pareto_rbf_svm,
        param_grid = rbf_svm_params,
        **gs_common_params
    )

    scaled_rbf_svm_gs = GridSearchCV(
        estimator = scaled_rbf_svm,
        param_grid = rbf_svm_params,
        **gs_common_params
    )


    #linear models
    linear_common_params = {
        'max_iter':100000,
        'tol':1e-7
    }

    lasso = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("est", sklin.Lasso(**linear_common_params),)
        ]
    )

    ridge = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("est", sklin.Ridge(**linear_common_params),)
        ]
    )

    elastic = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #('log',sklearn.preprocessing.FunctionTransformer(logabs)),
            ("scaling",CorrectedScaler(pareto=False)),
            ("est", sklin.ElasticNet(**linear_common_params),)
        ]
    )

    lasso_gs = GridSearchCV(
        estimator = lasso,
        param_grid = lasso_params,
        **gs_common_params
    )

    ridge_gs = GridSearchCV(
        estimator = ridge,
        param_grid = ridge_params,
        **gs_common_params
    )

    elastic_gs = GridSearchCV(
        estimator = elastic,
        param_grid = elastic_params,
        **gs_common_params
    )

    forest = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #("scaling",CorrectedScaler(pareto=False)),
            ("forest", skens.RandomForestRegressor(),)
        ]
    )

    forest_gs = GridSearchCV(
        estimator = forest,
        param_grid = forest_est_params,
        **gs_common_params
    )
    
    xg = imblearn.pipeline.Pipeline(
        steps=pre_steps + [
            #("scaling",CorrectedScaler(pareto=False)),
            ("forest", xgboost.XGBRegressor(nthread=1),)
        ]
    )

    xg_gs = GridSearchCV(
        estimator = xg,
        param_grid = xgboost_est_params,
        **gs_common_params
    )

    models = {
        'OLS': ols_gs,
        'PLS': unscaled_pls_gs,
        'Linear SVR': linear_svm_gs,
        'Pareto Poly SVR': pareto_poly_svm_gs,
        'Scaled Poly SVR': scaled_poly_svm_gs,
        'RBF SVR': rbf_svm_gs,
        'Pareto RBF SVR': pareto_rbf_svm_gs,
        'Scaled RBF SVR': scaled_rbf_svm_gs,
        'LASSO': lasso_gs,
        'Ridge': ridge_gs,
        'ElasticNet': elastic_gs,
        'Random Forest': forest_gs,
        #'XGBoost':xg_gs
    }
    return models

class CollinearSelectorMax(CollinearFeatureSelector):
    def fit(self,X,y=None):
        X = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            force_all_finite=True,
            copy=True,
        )
        if y is not None:  # Just so this passes sklearn api checks
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_2d=True,
                force_all_finite=True,
                y_numeric=False,
            )
        self.n_features_in_ = X.shape[1]

        (
            selected_features,
            cluster_id_to_feature_ids,
            _,
        ) = InspectData.cluster_collinear(
            X=X,
            feature_names=None,  # We use indices here not names to be consistent with sklearn.feature_selection.SelectFromModel
            t=self.t,
            display=False,  # We can change this to False so we don't have these plots
        )
        self.is_fitted_ = True

        self.feature_importances_ = np.zeros(X.shape[1], dtype=np.float64)
        if self.minimize_label_entropy is True:
            pass
        else:
            # Select a random feature from each cluster
            # Find the member with the largest variance in the cluster
            stds = X.std(axis=0)
            np.random.seed(self.seed)
            for cluster,feature_idcs in cluster_id_to_feature_ids.items():
                max_std = 0
                best_idx = -1
                best_idx = feature_idcs[0]
#                 for feature_idx in feature_idcs:
#                     feature_std = stds[feature_idx]
#                     if feature_std > max_std:
#                         best_idx = feature_idx
#                         max_std = feature_std
                self.feature_importances_[best_idx] = 1
#             for cluster in cluster_id_to_feature_ids.keys():
                
#                 feature_idx = cluster_id_to_feature_ids[cluster][
#                     np.random.randint(
#                         0, high=len(cluster_id_to_feature_ids[cluster])
#                     )
#                 ]
#                 self.feature_importances_[feature_idx] = 1

        return self

class CoefficientofVariationThreshold(sklearn.base.BaseEstimator,sklearn.base.TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        mean = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0, ddof=1)
        # Avoid division by zero for zero-mean features
        self.cv_ = np.divide(std, np.abs(mean), out=np.zeros_like(std), where=mean!=0)
        return self
        
    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.cv_ >= self.threshold]