import matplotlib.pyplot as plt
from .stats import compute_fit,compute_vals

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

def cvs_plots(study_data,cvs_dict,kfold_repeats,color='r'):
    x,y = study_data
    
    yt = np.squeeze(y.values)
    #cvs_dict = this_study.cvs_dict
    #n_repeats = this_study.kfold_repeats

    nplots = len(cvs_dict)
    fig,axs = plt.subplots(nplots,2,figsize=(8,2*nplots+0.5),squeeze=False)

    for (name,cvs),ax_cal,ax_res in zip(cvs_dict.items(),axs[:,0],axs[:,1], ):

        coefs,covar,fit_err = compute_fit(yt,cvs,kfold_repeats)

        plot_vals(ax_cal,yt,cvs,color=color)
        plot_vals(ax_res,yt,cvs,resid=True,color=color)

        plot_fit(ax_cal,coefs,covar,fit_err)
        ax_cal.text(0.05,0.9,name,transform=ax_cal.transAxes)
    return fig,axs

def calibration_plots(y,predicted,axs,size=1,color='r',label=None):
    #x,y = study_data
    #predicted = study_predict(x)
    
    #nplots = len(predicted)
    #fig,axs = plt.subplots(nplots,2,figsize=(8,2*nplots))
    
    for (name,pred),ax_cal,ax_res in zip(predicted.items(),axs[:,0],axs[:,1], ):
        ax_cal.plot(
            y,pred,
            ls='none',marker='o',markersize=size,color=color,
            label=label
        )

        cal_line_limits = ax_cal.get_xticks()[[0,-1]]
        ax_cal.plot(cal_line_limits,cal_line_limits,color='k',lw=2,zorder=-1)
        ax_cal.set_xlim(cal_line_limits)
        ax_res.plot(
            y,pred[:,None]-y.values,
            ls='none',marker='o',markersize=size,color=color
        )
        ax_cal.text(0.05,0.9,name,transform=ax_cal.transAxes)