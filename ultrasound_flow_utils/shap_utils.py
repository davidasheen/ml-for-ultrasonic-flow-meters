import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

shap_title_string = "SHAP values for {num} Dataset: {segm} set"

def get_median(df):
    return pd.DataFrame(
        np.median(df,axis=0)[None,:,],
        columns=df.columns
    )
    

def get_shaps(study_obj,model_name,median):
    def f(x):
        return study_obj.best_models[model_name].predict(x)
    ex1 = shap.Explainer(f,median)
    shap_train = ex1(study_obj.train[0],max_evals=2000)
    shap_values = ex1(study_obj.test[0],max_evals=2000)
    
    return ex1,shap_train,shap_values

def beeswarm_plot(
    shaps,
    dataset,
    segment,model_name,shap_plot_kw={}):
    title = shap_title_string.format(
        num = dataset,
        segm=segment,
    )

    shap.plots.beeswarm(
        shaps,
        order=shaps.abs.max(0),
        **shap_plot_kw
    )
    fig,ax = plt.gcf(),plt.gca()
    ax.set_title(title)
    ax.text(0.1,0.1,model_name,ha='left',va='bottom',transform=ax.transAxes)
    new_labels = [
        (
            label.get_text()
            .replace('media_','mean ')
            .replace('maximo_','max ')
            .replace('desvio_padrao_','std dev ')
        )
        for label in ax.get_yticklabels()
    ]
    t = ax.set_yticklabels(new_labels)