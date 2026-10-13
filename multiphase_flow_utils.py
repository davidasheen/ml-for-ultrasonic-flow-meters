import numpy as np
import pandas as pd
import imblearn
import sklearn
import xgboost

from itertools import combinations

import matplotlib.pyplot as plt

import sklearn.decomposition as skdecomp
import sklearn.ensemble as skens
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.cross_decomposition as skpls
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold,StratifiedKFold,KFold
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pychemauth.preprocessing.scaling import CorrectedScaler
from pychemauth.preprocessing.feature_selection import CollinearFeatureSelector
from pychemauth.eda.explore import InspectData

def robust_r2(true,pred):
    msd_res = np.median(
        (pred - true)**2
    )
    msd_tot = np.median(
        (pred - np.median(pred))**2
    )
    return 1 - msd_res/msd_tot

score_metrics = dict(
    mae=sklearn.metrics.mean_absolute_error,
    rmse=sklearn.metrics.root_mean_squared_error,
    #r2=robust_r2,
    r2=sklearn.metrics.r2_score,
    mape=sklearn.metrics.mean_absolute_percentage_error,
    medae=sklearn.metrics.median_absolute_error,
    maxe=sklearn.metrics.max_error,
)

#process vol and flow vel columns
process_qvol_col = 'Q Vol ref ([m³/h])'
process_vel_col = 'Vel (m/s)'
process_qv_ratio = 'Q Vol / Vel (m²)'
hour2sec = 3600

process_pres_name = 'Pressão absoluta ([kPa])'
process_pres_translate = 'Absolute pressure ([kPa])'

qmeter_string = 'QMeter'
qflow_string = 'QFlow'
flowvela_string = 'FlowVelA'
flowvelb_string = 'FlowVelB'
flowvelc_string = 'FlowVelC'
flowveld_string = 'FlowVelD'

def strip_diagnostic_flow_rate(xdata):
    
    qmeter_cols = [x for x in xdata.columns.values if qmeter_string in x]
    qflow_cols = [x for x in xdata.columns.values if qflow_string in x]
    flowvela_cols = [x for x in xdata.columns.values if flowvela_string in x]
    flowvelb_cols = [x for x in xdata.columns.values if flowvelb_string in x]
    flowvelc_cols = [x for x in xdata.columns.values if flowvelc_string in x]
    flowveld_cols = [x for x in xdata.columns.values if flowveld_string in x]
    
    cols_to_drop = qmeter_cols + qflow_cols + flowvela_cols + flowvelb_cols + flowvelc_cols + flowveld_cols
    #print(qmeter_cols)
    if not qmeter_cols == []:
        data_position = 0
        for flowvel_col in [
            flowvela_cols,
            flowvelb_cols,
            flowvelc_cols,
            flowveld_cols
        ]:
            for meter,flow in zip(qmeter_cols,flowvel_col):
                print(meter,flow)

                #print(meter.split()[0].split('_')[-1])
                try:
                    meter_stat,meter_name, = meter.split()[0].split('_')
                    flow_stat,flow_name, = flow.split()[0].split('_')
                except:
                    meter_stat1,meter_stat2,meter_name, = meter.split()[0].split('_')
                    flow_stat1,flow_stat2,flow_name, = flow.split()[0].split('_')
                    meter_stat = '_'.join((meter_stat1,meter_stat2))
                    flow_stat = '_'.join((flow_stat1,flow_stat2))

                print(meter_name,meter_stat,flow_name,flow_stat)

                new_col_name = '_'.join((meter_stat,meter_name)) + '/' + flow_name + ' (m2)'
                print(new_col_name)
                xdata.insert(data_position,new_col_name,xdata[meter]/xdata[flow])
                data_position+=1
                #xdata[new_col_name] = xdata[meter]/xdata[flow]

                xdata = xdata.replace([np.inf, -np.inf], np.nan)#.dropna(axis=0,how='any')
        xdata = xdata.drop(columns = cols_to_drop)
    return xdata

def read_and_split_file(
    file,ydata_col,
    drop_cols=None,drop_rows=None,
    products=False,triples=False,quads=False,
    noise=True,
):
    'Reads the flow data from an excel file and splits into the x and y data'
    df = pd.read_excel(file,)
    print(ydata_col)
    
    if drop_rows:
        df = df.drop(drop_rows)
    
    ydata = df[[ydata_col]]
    xdata = df.drop(columns=[ydata_col])
    if drop_cols:
        print(xdata.columns)
        xdata = xdata.drop(columns=drop_cols)
    
    if process_pres_name in xdata:
        xdata = xdata.rename(columns={process_pres_name:process_pres_translate})
    
    #remove vol flow rate and flow speed from process data set and
    #replace with ratio (effectively, cross-sectional area)
    print(process_qvol_col in xdata)
    print(process_vel_col in xdata)
    if process_qvol_col in xdata and process_vel_col in xdata:
        xdata[process_qv_ratio] = xdata[process_qvol_col]/xdata[process_vel_col]/hour2sec
        #print(xdata[process_qv_ratio])
        xdata = xdata.drop(columns=[process_qvol_col,process_vel_col])
    
    #remove vol flow rate and flow speed from process data set and
    #replace with ratio (effectively, cross-sectional area
    xdata = strip_diagnostic_flow_rate(xdata)
    
    
    
    #remove nan values from xdata and ydata
    xdata_nans_mask = xdata.isna().any(axis=1).values
    xdata=xdata[~xdata_nans_mask]
    ydata=ydata[~xdata_nans_mask]
    
    if products:
        products_frame = pd.DataFrame(
            #{f"{col1}x{col2}":xdata[col1]*xdata[col2] for col1,col2 in combinations(xdata.columns,2)}
            {f"({col1})²":xdata[col1]*xdata[col1] for col1 in xdata.columns}
        )
        triples_frame = None
        quads_frame = None
        if triples:
            triples_frame = pd.DataFrame(
                {f"({col1})³":xdata[col1]*xdata[col1]*xdata[col1] for col1 in xdata.columns}
            )
        if quads:
            quads_frame = pd.DataFrame(
                {f"({col1})⁴":xdata[col1]*xdata[col1]*xdata[col1]*xdata[col1] for col1 in xdata.columns}
            )
            
        xdata = pd.concat((xdata,products_frame,triples_frame,quads_frame,),axis=1)
    
    if noise:
        noise_mean = 0
        noise_std = 1e-12
        noise_data = np.random.normal(noise_mean,noise_std,xdata.shape)
        xdata = xdata + noise_data
    
    return xdata,ydata  

def strip_constant(df):
    'Identifies columns in a DataFrame with zero standard deviation and removes them'
    
    stdev = df.std()
    drop_cols = stdev[stdev==0].index
    
    return df.drop(columns=drop_cols)

def prune_threshold(df,thresh=0.05):
    'Identifies columns with a coefficient of variation below a certain threshold and removes them'
    
    stdev = df.std()
    mn = df.mean()
    
    cofv = np.abs(stdev/mn)
    below_thresh = cofv < thresh
    
    drop_cols = stdev[below_thresh].index
    
    return df.drop(columns=drop_cols)

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

def plot_fit(ax,coefs,covar,fit_err,num=51):
    
    cal_line_limits = ax.get_xticks()[[0,-1]]
    fit_x = np.linspace(*cal_line_limits,num)
    
    fit_vals,fit_varis = compute_vals(fit_x,coefs,covar,fit_err)
    fit_95 = np.sqrt(fit_varis)*2
    
    ax.plot(fit_x,fit_vals,color='k',ls='--',)
    ax.plot(fit_x,fit_vals+np.sqrt(fit_varis)*2,color='k',ls=':')
    ax.plot(fit_x,fit_vals-np.sqrt(fit_varis)*2,color='k',ls=':')
    
    ax.plot(cal_line_limits,cal_line_limits,color='k',lw=2)
    ax.set_xlim(cal_line_limits)
    
    return

def plot_vals(ax,yt,cvs,resid=False,color='r'):
    cv_eff = np.squeeze(
        np.stack(cvs,axis=-1)
    )
    
    if resid:
        cv_eff = cv_eff - yt[:,None]
    
    ax.plot(
        yt,cv_eff,
        ls='none',marker='o',markersize=1,color=color
    )
    
    return

def plot_cvs(cvs_dict,yt):
    nplots = len(cvs_dict)
    fig,axs = plt.subplots(nplots,figsize=(6,4*nplots))
    
    for (name,cvs),ax in zip(cvs_dict.items(),axs.flatten() ):
        #print(cvs)
        for cv in cvs:
            #print(len(cv))
            ax.scatter(
                yt,cv
            )
        ax.plot(
            (-4,6),
            (-4,6),
        )
        ax.text(0.05,0.9,name,transform=ax.transAxes)
    return


class flow_study(object):
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
    var_step = [(
        'var',CoefficientofVariationThreshold(threshold=thresh)
    )]
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