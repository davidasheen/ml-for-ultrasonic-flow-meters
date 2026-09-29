import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import imblearn
import sklearn
import pickle

import sklearn.decomposition as skdecomp
import sklearn.ensemble as skens
import sklearn.svm as sksvm
import sklearn.linear_model as sklin

from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold,StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

import multiphase_flow_utils

from pychemauth.classifier.simca import SIMCA_Authenticator
from pychemauth.manifold.elliptic import EllipticManifold_Authenticator
from pychemauth.classifier.plsda import PLSDA
#from pychemauth.analysis.compare import Compare
from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.preprocessing.feature_selection import CollinearFeatureSelector,JensenShannonDivergence

from pychemauth.classifier.osr import OpenSetClassifier

from pychemauth_compare_ext import Compare_reg

from pychemauth.eda.explore import InspectData
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
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

from argparse import ArgumentParser

def get_parser():

    parser = ArgumentParser(
        prog="lus_run_loader",
        description="run all the ultrasound models",
    )

    parser.add_argument("--file", "-f", default="", type=str, help="data file name")
    parser.add_argument("--n_jobs", "-n", default=16, type=int, help="number of jobs")
    parser.add_argument("--hcathresh", "-t", default=0.5, type=float, help="threshold for hierarchical cluster analysis")
    return parser

#def main(args=None):
parser = get_parser()

options = (
    parser.parse_args() #if args is None else parser.parse_args(args)
)

print(options)
print(options.file)

file=options.file
n_jobs=options.n_jobs
t=options.hcathresh

print(n_jobs)

study_pickle_file = file.split('.')[0] + f'_velratio_vs_{t}_study.pkl'
print(study_pickle_file)
ydata_col = 'Erro ([%])'

this_study = multiphase_flow_utils.flow_study(
    file=file,
    ydata_col=ydata_col,
    train_size=0.8,
    split_random_state=42,
    kfold_repeats=40,
    n_jobs=n_jobs,
    log_scale=True,
)

collinear_kw = dict(t=t,seed=42,minimize_label_entropy=False)
selector = CollinearSelectorMax(**collinear_kw)

xdata,ydata = this_study.data

selector.fit(xdata)

xnew = selector.transform(xdata)

xtrain,xtest,ytrain,ytest = train_test_split(
    xnew,ydata,
    train_size=this_study.train_size,
    random_state=this_study.split_random_state
)
#xtrain=xtrain.values

this_study.data = xnew,ydata
this_study.train = xtrain,ytrain
this_study.test = xtest,ytest



this_study.fit_and_cv(n_jobs=n_jobs)

with open(study_pickle_file,'wb') as f:
    print(study_pickle_file)
    pickle.dump(this_study,f,protocol=4)

