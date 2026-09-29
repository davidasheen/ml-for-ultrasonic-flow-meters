import matplotlib.pyplot as plt
from pychemauth_compare_ext import Compare_reg,make_boxplot,order_results,results_perform

def plot_and_analyze(
    study_obj,
    score='r2',
    models=['Random Forest'],
    reverse=True,
    label=r'$R^2$ score'
):
    
    scores_dict = study_obj.cv_scores[score]
    scores_dict = {mdl:scores_dict[mdl] for mdl in models}
    
    make_boxplot(scores_dict,study_obj.kfold_repeats,reverse=reverse)
    plt.gca().set_xlabel(label)
    
    return

def get_best_models(
    study_obj,
    score='r2',
    models=['Random Forest'],
    alpha=0.05,
    reverse=False
):
    scores_dict = study_obj.cv_scores[score]
    scores_dict = {mdl:scores_dict[mdl] for mdl in models}
    
    perfs = results_perform(scores_dict,study_obj.kfold_repeats,alpha=alpha,reverse=reverse)
    best_model = perfs[0][0]
    good_models = []
    for perf in perfs:
        if not perf[-1]: good_models += [perf[0]]
        
    return best_model,good_models