

import sklearn.metrics
from sklearn.model_selection import train_test_split

from .stats import fit_models,generate_kfolds,extract_best_estimator,cross_validate,compute_scores
from .data import strip_diagnostic_flow_rate,read_and_split_file,prune_threshold,strip_constant
from .utils import generate_models

class FlowStudy(object):
    def __init__(
        self,
        file,ydata_col,
        drop_xdata_cols=None,
        drop_outlier_rows=None,
        prune=False,prune_thresh=0.1,
        products=False,triples=False,quads=False,
        variable_selection=False,variable_selector=None,
        log_scale=False,
        train_size=0.8,
        split_random_state=None,
        kfold_random_state=42,
        kfold_repeats=5,
        n_jobs=8
    ):
        self.file = file
        self.ydata_col=ydata_col
        self.drop_xdata_cols=drop_xdata_cols
        self.drop_outlier_rows=drop_outlier_rows
        self.train_size = train_size
        self.split_random_state=split_random_state
        self.kfold_repeats=kfold_repeats
        self.kfold_random_state=kfold_random_state
        self.cv_scores = {}
        
        self.prepare_data(
            prune=prune,
            prune_thresh=prune_thresh,
            products=products,triples=triples,quads=quads,
            selection=variable_selection,
            selector=variable_selector,
        )
        
        self.kfolds = generate_kfolds(
            n_repeats=self.kfold_repeats,
            random_state=self.kfold_random_state
        )
        
        xdata,ydata = self.data
        
        self.models = generate_models(
            xdata,
            log_scale=log_scale,
            thresh=prune_thresh,
            selection=variable_selection,
            selector=variable_selector,
            n_jobs=n_jobs
        )
        #self.selection = variable_selection
        #self.selector = variable_selector
        
        return

    def prepare_data(
        self,
        prune=False,prune_thresh=0.1,
        products=False,triples=False,quads=False,
        selection=False,selector=None
    ):
        xdata,ydata = read_and_split_file(
            self.file,self.ydata_col,
            drop_cols=self.drop_xdata_cols,
            drop_rows=self.drop_outlier_rows,
            products=products,triples=triples,quads=quads,
        )
        xdata.columns = xdata.columns.str.replace('[','')
        xdata.columns = xdata.columns.str.replace(']','')
        xdata = strip_constant(xdata)
        if prune: 
            xdata = prune_threshold(xdata,thresh=prune_thresh)
            print('pruning')
            print(prune_thresh)
        #if selection:
        #    selector.fit(xdata)
        #    xdata = selector.transform(xdata)

        xtrain,xtest,ytrain,ytest = train_test_split(
            xdata,ydata,
            train_size=self.train_size,
            random_state=self.split_random_state
        )
        #xtrain=xtrain.values
        
        self.data = xdata,ydata
        self.train = xtrain,ytrain
        self.test = xtest,ytest

        return #xdata,ydata,xtrain,xtest,ytrain,ytest,
    
    def fit_and_cv(self,n_jobs=8):
        print('Beginning model training')
        
        fit_models(
            self.models,
            *self.train
        )
        print('Model training complete, beginning nested cross-validation')
        
        self.best_models = extract_best_estimator(self.models)
        
        self.cvs_dict = cross_validate(self.best_models,*self.train,self.kfolds,n_jobs=n_jobs)
                
        return
    
    def all_predict(self,x):
        return {
            k:model.predict(x) 
            for k,model in self.models.items()
        }
        
    
    def cv_score(self,score_name='r2',score_fun=sklearn.metrics.r2_score):
        print(score_name)
        self.cv_scores[score_name]= compute_scores(
            self.cvs_dict,
            *self.train,
            self.kfolds,
            score_fun=score_fun
        )
        
        return
    
    def all_score(
        self,
        xt,yt,
        score_name='r2',score_fun=sklearn.metrics.r2_score
    ):
        predictions = self.all_predict(xt)
        scores = {
            k:score_fun(yt,ypred,) for k,ypred in predictions.items()
        }
        
        return {score_name:scores}