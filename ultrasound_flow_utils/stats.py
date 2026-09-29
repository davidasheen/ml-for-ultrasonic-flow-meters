import numpy as np
import sklearn.metrics
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold,StratifiedKFold,KFold
from sklearn.model_selection import train_test_split,cross_val_predict

def fit_models(models,xt,yt):
    for name,model in models.items():
        print(name)
        model.fit(xt,yt.iloc[:,0])
    return

def generate_kfolds(n_repeats=5,random_state=42):
    kfolds = [
        KFold(n_splits=7,shuffle=True,random_state=random_state+n)
        for n in np.arange(n_repeats)
    ]
    return kfolds

def extract_best_estimator(models):
    best_models = {name:model.best_estimator_ for name,model in models.items()}
    return best_models

def cross_validate(models,xt,yt,kfolds,n_jobs=8):
    cvs_dict = {}
    
    for name,model in models.items():
        model_cvs = []
        for kf in kfolds:
            cvs = cross_val_predict(
                model,
                xt,yt.iloc[:,0],
                cv=kf,
                n_jobs=n_jobs
            )
            model_cvs += [cvs]
        cvs_dict[name] = model_cvs
    
    return cvs_dict

def compute_scores(cvs_dict,xt,yt,kfolds,score_fun=sklearn.metrics.r2_score):
    scores_dict = {}
    for name,cvs in cvs_dict.items():
        model_scores = []
        for cv,kf in zip(cvs,kfolds,):
            scores = []
            for train,test in kf.split(xt,yt,):
                this_cv = cv[test]
                this_ytest = yt.iloc[test].iloc[:,0]
                score = score_fun(this_ytest,this_cv)

                model_scores += [score]
        scores_dict[name] = model_scores
    return scores_dict

def compute_fit(yt,cvs,n_repeats,deg=1):
    
    y_eff = np.tile(yt,reps=n_repeats)
    cv_eff = np.squeeze(np.concatenate(cvs))
    
    coefs,covar = np.polyfit(y_eff,cv_eff,deg=deg,cov=True)
    
    vals_fit = np.polyval(coefs,y_eff)
    
    fit_err = np.square(y_eff - vals_fit).sum()/(y_eff.shape[0] - deg - 1)
    
    return coefs,covar,fit_err

def compute_vals(x,coefs,covar,fit_err):
    onegrid = np.ones_like(x)
    xnew = np.stack((x,onegrid),axis=0)
    
    preds = (coefs.T @ xnew).T
    varis = np.sum(xnew*np.dot(covar.T,xnew),axis=0) + fit_err
    
    return preds,varis