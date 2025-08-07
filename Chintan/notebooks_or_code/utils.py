from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib.text as mtext
import seaborn as sns
import numpy as np
from math import sqrt, ceil
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import scipy

def get_validation_plot(x, y, kde=True, suptitle=None, validation=True, theil_slope=False, log_norm=False, ax_val=None, title_val="", xlabel_val="", ylabel_val="", color_val="blue", metrics=['r2','mae','nrmse','mape','mdsa','slope', 'intercept','bias'], trendline=True, reference_line=True, residuals=False, ax_res=None, title_res="", xlabel_res="", color_res="black", res_log=False, legend_loc='upper left',reference_line_legend=True, marker_size=25, marker_border='none', category=None, separator=None, separator_legend=False, min_threshold=-100, grid=0.5):
    """
    x: true/observed value, y: predicted value
    x and y are array like, in the shape (n); where n=no of samples
    This function returns a plot with the validation fit, regression line, reference line and title for the given data
    """
    # log_norm = False
    # verify args
    if ax_val is not None and not validation:
        return "Invalid arguments - validation flag False but validation axis specified"
    elif ax_res is not None and not residuals:
        return "Invalid arguments - residuals flag False but residuals axis specified"
    elif validation and ax_val is not None and residuals and ax_res is None:
        return "Invalid arguments - residuals axis not specified but validation axis specified"
    elif residuals and ax_res is not None and validation and ax_val is None:
        return "Invalid arguments - validation axis not specified but residuals axis specified"
    
    # restrict values smaller than min threshold
    if min_threshold is not None:
        x[x<min_threshold] = min_threshold
        y[y<min_threshold] = min_threshold

    x_linear = x
    y_linear = y
    if log_norm:
        x, y = np.log10(x), np.log10(y)
        x[x<0], y[y<0] = 0.00001, 0.00001

    
    # fit linear model to get R2 value    
    lm = scipy.stats.linregress(x=x, y=y)
    slope, intercept = lm.slope, lm.intercept

    # # check for thiel slope flag # Thiel currently is only defined for univariate fitting. Researching ways to derive thiel slopes for multi-variate fitting
    # if theil_slope:
    #     slope, intercept, low_slope, high_slope = stats.theilslopes(y, x, 0.95, method='separate')
    #     # calculate thiel R2
    #     theil_y_pred = slope*y + intercept
    #     lm_theil = scipy.stats.linregress(x=x, y=theil_y_pred)

    # calculate residuals
    residual_values = x - y

    # evaluation stats
    r2_value = round(lm.rvalue,2)
    rmse_value = round(root_mean_squared_error(x,y),2)
    nrmse_value = round(root_mean_squared_error(x,y)/(np.max(x)-np.min(x))*100,2)
    mape_value = round(mean_absolute_percentage_error(x,y)*100,2)
    # use mdsa is log-based so works only with positive values
    mdsa_value = round(100*(np.exp(np.median(np.abs(np.log(y/x))))-1),2) # from Morely, 2018 (https://doi.org/10.1002/2017SW001669) 
    signed_bias = round(100*(np.sign(np.median(np.log(y/x))))*(np.exp(np.median(np.abs(np.log(y/x))))-1) ,2) # signed systematic bias
    bias_linear =round((np.sum(y - x)/np.sum(x))*100,2)
    mae_value = round(mean_absolute_error(x,y),2)
    # mdsa_value = round((1-np.median(2*abs(x-y)/(abs(x)+abs(y))))*100,2) # this is a different error, need to find reference
    
    # load evaluation stats in a label hash
    label_hash = {'r2':rf"$R^2={r2_value}$", 'mae':rf"$MAE={mae_value}$", 'rmse':rf"$RMSE={rmse_value}$", 'nrmse':rf"$NRMSE={nrmse_value}$%", 'mape':rf"$MAPE={mape_value}$%", 'mdsa':rf"$MdSA={mdsa_value}$%", 'bias':rf"$\beta={signed_bias}$%", 'bias_linear': rf"$\beta={bias_linear}$%", 'slope':rf"$S={round(slope,2)}$", 'intercept':rf"$c={round(intercept,2)}$"}

    # create empty container plot, set fig and axes
    fig = None
    if (ax_val is None) and (ax_res is None):
        if validation and residuals:
            # 2 plots
            fig, ax = plt.subplots(1,2, figsize=(14,7))
            ax_val = ax[0]
            ax_res = ax[1]        
        elif validation and not residuals:
            fig, ax_val = plt.subplots()
        elif not validation and residuals:
            fig, ax_res = plt.subplots()
    if log_norm:
        # set log axis
        if ax_val: 
            ax_val.set_xscale('log')
            ax_val.set_yscale('log')
        # x = np.log(x)
        # y = np.log(y)
        # x[np.isnan(x)] = 0
        # x[np.isinf(x)] = 0
        # y[np.isnan(y)] = 0
        # y[np.isinf(y)] = 0

    # set super title 
    if suptitle is not None:
        fig.suptitle(suptitle)

    # validation scatter plot
    if validation:
        # prepare label

        label = '\n'.join((label_hash[i] for i in metrics))

        # add scatter plot and R2
        # scatterplot used for base scatterplot
         
        if separator == 'hue':
            if separator_legend:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, hue=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
            else:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, hue=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
                # legend_text = Line2D([], [], linestyle='none', label=label)
                # ax_val.legend(handles=[legend_text])

        elif separator == 'style':
            if separator_legend:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, style=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
            else:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, hue=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
                # legend_text = Line2D([], [], linestyle='none', label=label)
                # ax_val.legend(handles=[legend_text])

        elif separator == 'uncertainty':
            pass
            # to be developed
            ax_val.scatter(x_linear, y_linear, c=category, cmap="RdYlGn_r", edgecolor='k', alpha=0.8)
            
        elif separator == None:
            sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, ax=ax_val, color=color_val, s=marker_size, legend=False).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
        else:
            return "Invalid argument for separator - must be either \"hue\" or \"style\"!"
        # add kde contour lines
        if kde:
            sns.kdeplot(x=x_linear, y=y_linear, color='black', log_scale=log_norm, fill=False, ax=ax_val, alpha=0.2)

        # include theil slopes
        if theil_slope:
            ax_val.plot(x, intercept + slope*x, 'k-.')
            ax_val.plot(x, intercept + low_slope*x, 'r--')
            ax_val.plot(x, intercept + high_slope*x, 'r+')
        
        # add trendline
        if trendline:
            reference_line_opacity = 0.6
            # sns.regplot(x=x, y=y, ci=95, label=label, ax=ax_val, color=color_val,).set(title=f"{title_val}\nN = {len(x)}", 
            #                                                                      xlabel=xlabel_val, 
            #                                                                      ylabel=ylabel_val)
            # regplot only used to add trendline
            # sns.regplot(x=x, y=y, ci=95, scatter=False, ax=ax_val, color='black',)

            # # extend both axis equally to match min and max
            # lims = [
            # np.min([ax_val.get_xlim(), ax_val.get_ylim()]),   # min of both axes
            # np.max([ax_val.get_xlim(), ax_val.get_ylim()]),   # max of both axes
            # ]   
        
            # # equal axes length and add 1:1 line
            # ax_val.set_xlim(lims)
            # ax_val.set_ylim(lims)

            if log_norm:
                min_val, max_val = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
                x_fit = np.logspace(min_val, max_val, 100)
                y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
                ax_val.plot(x_fit, y_fit, color=color_val)
            else:
                xmin, xmax = ax_val.get_xlim()
                xextended = np.array([xmin, xmax])
                prediction_fit = slope*xextended + intercept
                ax_val.plot(xextended, prediction_fit, color=color_val)
        else:
            reference_line_opacity = 1.0
        
        if reference_line:
            # for 1:1 line
            lims = [
            np.min([ax_val.get_xlim(), ax_val.get_ylim()]),   # min of both axes
            np.max([ax_val.get_xlim(), ax_val.get_ylim()]),   # max of both axes
            ]   
        
            # equal axes length and add 1:1 line
            ax_val.set_xlim(lims)
            ax_val.set_ylim(lims)
            if reference_line_legend == True:
                # ref_label = '1:1'
                ref_line_handle = Line2D([], [], alpha=reference_line_opacity, color='black', linestyle='--', label='1:1')
            else:
                ref_line_handle = Line2D([], [], label=None)
            if log_norm:
                ax_val.axline((1, 1), (10, 10), alpha=reference_line_opacity, zorder=0, linestyle='dashed', color='black', label = '1:1')
            else:
                ax_val.axline((0,0), slope=1, alpha=reference_line_opacity, zorder=0, linestyle='dashed', color='black')

        # add legend and grid
        if not separator_legend:
            if reference_line_legend == True:
                # add metrics
                legend_text_metrics = Line2D([], [], color='none', label=label)

                # ref_label = '1:1'
                ref_line_handle = Line2D([], [], alpha=reference_line_opacity, color='black', linestyle='--', label='1:1')
                ax_val.legend(handles=[legend_text_metrics, ref_line_handle], loc=legend_loc, frameon=True, handletextpad=0.2, handlelength=1)
            else:
                legend_metrics_handle = label
                ax_val.legend(
                    handles=[legend_metrics_handle],
                    labels=[label],
                    handler_map={legend_metrics_handle: NoPaddingHandler()},
                    loc=legend_loc,
                    frameon=True,
                    handletextpad=0.2, 
                    handlelength=1
                )
                # ax_val.legend(handles=[legend_metrics_handle,], loc=legend_loc, frameon=True)

            # legend_text_metrics = Line2D([], [], linestyle='none',)
            # handles, labels = ax_val.get_legend_handles_labels()
            # handles.append(legend_text_metrics)
            # labels.append(label)
            # handles = handles[1:] + [handles[0]]
            # labels = labels[1:] + [labels[0]]
            # ax_val.legend(
            #     handles=handles,
            #     labels=labels,
            #     handler_map={tuple: HandlerTuple(ndivide=None)},
            #     loc=legend_loc,
            #     frameon=True
            # )
            # ax_val.legend(handles=[legend_text], loc=legend_loc)
        else:
            ax_val.legend(loc=legend_loc, handletextpad=0.2, handlelength=1)
        ax_val.grid(alpha=grid)
        # plt.show()

    # residuals' histogram
    if residuals:
        print('set residuals')
        sns.histplot(residual_values, log_scale=log_norm, color=color_res, kde=(not log_norm), ax=ax_res).set(title=f"{title_res}\nN={len(residual_values)}", xlabel=xlabel_res)
    # print('***********plotting finished***********')

    return fig, {'r2': r2_value, 'rmse':rmse_value, 'nrmse': nrmse_value, 'mape':mape_value, 'mae':mae_value, 'mdsa':mdsa_value, 'bias':signed_bias, 'bias_linear': bias_linear, 'residuals':residual_values, 'slope': round(slope,2), 'intercept': intercept}
