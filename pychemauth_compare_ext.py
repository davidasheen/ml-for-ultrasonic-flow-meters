from pychemauth.analysis.compare import Compare
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import math

iqr_correction = 1.349

def order_results(results,reverse=True,ignore=np.nan):
    if np.isnan(ignore):
        key = lambda k: np.median(
            np.asarray(results[k])[~np.isnan(results[k])]
        )
    else:
        key = lambda k: np.median(
            np.asarray(results[k])[np.asarray(results[k]) != ignore]
        )

    order = sorted(results, key=key, reverse=reverse)
    return order

def results_perform(results,n_repeats,alpha,reverse=True,asymmetric=False,ignore=np.nan):
    order = order_results(results,reverse=reverse)
    #print(order)
    
    #iqr_correction = 1.349
    performances = []
    for name in order:
        #print(name)
        scores = results[name]
        mdn = np.median(scores)
        quantiles = np.quantile(scores,[0.25,0.75])
        #compute interquartile range
        iqr = (quantiles[1] - quantiles[0]) 
        #compute a standard deviation from the interquartile range
        if not asymmetric:
            iqr = iqr/iqr_correction
        else:
            iqr = quantiles[:,None]
            iqr = np.abs(iqr - mdn)
            #iqr[iqr<0] = 0


        performances += [[name,mdn,iqr,False]]
    for i in range(1, len(performances)):
        #print(order[0],', ',order[i])
        p = Compare_reg.corrected_t(
            results[order[0]],
            results[order[i]],
            n_repeats,
            ignore=ignore,
        )
        if not reverse: p = 1-p
        # Do we REJECT H0 (that pipelines perform the same)?
        performances[i][-1] = p < alpha

    return performances

def perf_colors(performances,cmap='viridis',):
    colors = []
    for i, p in enumerate(performances):
        if not p[-1]:  # REJECT H0, so pipeline 1 DOES outperform this one
            color = matplotlib.colormaps[cmap](
                (1.0 + i) / len(performances)
            )
            hue = 1.0
        else:
            color = "gray"
            hue = 0.5
        colors += [(color,hue)]
    return colors

def make_boxplot(results,kfold_repeats,reverse=True,asymmetric=True,ignore=np.nan,ax=None,**kwargs):
    order = order_results(results,reverse=reverse)
    
    if ax is None:
        fig,ax = plt.subplots()

    h = ax.boxplot(
        [results[name] for name in order][::-1],
        #[scr for name,scr in scores_dict.items()],
        tick_labels=order[::-1],
        vert=False,
        **kwargs
    )
    
    #position of colored boxes
    axlims = ax.get_xlim()
    left = axlims[0]
    right = axlims[1]
    width = right - left
    #bottom=0.5 <-- note that bottom will be different for each box
    height = 1
    perfs = results_perform(
        results,
        kfold_repeats,
        alpha=0.05,
        reverse=reverse,
        asymmetric=asymmetric,
    )

    colors = perf_colors(perfs)

    for i,color in enumerate(colors[::-1]):
        c,h = color
        bottom = 0.5+i
        rect = matplotlib.patches.Rectangle(
            (left,bottom),width,height,color=c,alpha=h
        )
        ax.add_patch(rect)
    return

class Compare_reg(Compare):
    
    @staticmethod
    def corrected_t(
        scores1, scores2, n_repeats, ignore
    ):
        '''
        Modified to use robust statistics
        '''
        scores1, scores2, mask = Compare._check_scores(
            scores1=scores1, scores2=scores2, ignore=ignore
        )

        k_fold = len(scores1) // int(n_repeats)
        n = k_fold * n_repeats
        assert n == len(scores1), "scores must be divisible by n_repeats"

        rho = 1.0 / k_fold
        perf_diffs = scores1[mask] - scores2[mask]

        iqr = np.quantile(perf_diffs,[0.25,0.75])
        iqr = (iqr[1] - iqr[0])/iqr_correction

        corrected_t = (
            np.median(perf_diffs) - 0.0
        ) / np.sqrt(  # n -> sum(mask) to be consistent
            (1.0 / np.sum(mask) + rho / (1.0 - rho))
            * (iqr ** 2)
        )

        return 1.0 - scipy.stats.t.cdf(
            x=corrected_t, df=np.sum(mask) - 1
        )
    
    @staticmethod
    def visualize(
        results,
        n_repeats,
        alpha=0.05,
        ignore=np.nan,
        cmap='viridis',
        asymmetric=False,
    ):
        def perf(results,n_repeats,alpha):
            if np.isnan(ignore):
                key = lambda k: np.median(
                    np.asarray(results[k])[~np.isnan(results[k])]
                )
            else:
                key = lambda k: np.median(
                    np.asarray(results[k])[np.asarray(results[k]) != ignore]
                )

            order = sorted(results, key=key, reverse=True)

            #iqr_correction = 1.349
            performances = []

            for name in order:
            #for name,scores in scores_dict.items():
                #name = k
                print(name)
                scores = results[name]
                mdn = np.median(scores)
                quantiles = np.quantile(scores,[0.25,0.75])
                #compute interquartile range
                iqr = (quantiles[1] - quantiles[0]) 
                #compute a standard deviation from the interquartile range
                if not asymmetric:
                    iqr = iqr/iqr_correction
                else:
                    iqr = quantiles[:,None]
                    iqr = np.abs(iqr - mdn)
                    #iqr[iqr<0] = 0
                    

                performances += [[name,mdn,iqr,False]]


            for i in range(1, len(performances)):
                print(order[0],', ',order[i])
                p = Compare_reg.corrected_t(
                    results[order[0]],
                    results[order[i]],
                    n_repeats,
                    ignore=ignore,
                )
                # Do we REJECT H0 (that pipelines perform the same)?
                performances[i][-1] = p < alpha

            return performances

        performances = perf(results, n_repeats=n_repeats, alpha=alpha)
        print(performances)

        chart = plt.subplot(projection="polar")
        for i, p in enumerate(performances):
            if not p[-1]:  # REJECT H0, so pipeline 1 DOES outperform this one
                color = matplotlib.colormaps[cmap](
                    (1.0 + i) / len(performances)
                )
                hue = 1.0
            else:
                color = "gray"
                hue = 0.5
            _ = chart.barh(
                len(performances) - 1 - i,
                math.radians(p[1] * 360),
                label=p[0],
                xerr=np.radians(p[2] * 360),
                alpha=hue,
                color=color,
            )
            _ = chart.set_xticks(np.linspace(0, 0.9, 10) * 2.0 * np.pi)
            _ = chart.set_xticklabels(
                ["%.2f" % t for t in np.linspace(0.0, 0.9, 10)]
            )
            _ = chart.set_yticks([i for i in range(len(performances))])
            _ = chart.set_yticklabels([])
            _ = chart.set_title("Score +/- " + r"$\sigma$")
            _ = chart.legend(loc="best", bbox_to_anchor=(1.2, 0.5, 0.5, 0.5))

        return chart