"""
TODO.

Bring single_speed_kinematics_comparison (in clustering repo) and run_single_condition_stats (in dimred repo) together because they do pretty much the same

Pipeline_fatigue and pipeline_clustering are the same for 0D, multicondition (segments and speeds respectively) stats, so can also be merged together.
The only difference is the plotting, which can be done with if statements.

Then the second part of multispeed_kinematics_comparison is the same as run_SPM_ANOVA2onerm.
You need to rearrange the data in the clustering script to be in the same format as the fatigue script,
which is more aligned with SPM, and then you can run the same SPM code for both.


TODO. Test run_0D_ANOVA2onerm in clustering repo

"""

# %% Imports

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_utils.statistics import anova2onerm_0d_and_posthocs, compare_0D_contvar_indgroups_one_condition, write_0Dposthoc_statstr, write_0DmixedANOVA_statstr
from research_utils.vis import (
    visualise_0D_ANOVA2onerm,
    plot_0D_ANOVA2onerm_within_effect,
)
import seaborn as sns
from scipy import stats

# %% Functions

def run_demoanthrophys_two_groups_comparisons(datasheet, grouping_var, re_speeds, titles, **kwargs):
    """
    Compare demographics, anthropometrics and physiological variables between clusters.

    :return:
    """

    # Get kwargs
    ylabels = kwargs.get("ylabels", {var: "" for var in titles})
    group_names = kwargs.get("group_names", np.unique(datasheet[grouping_var]).tolist())
    group_colours = kwargs.get("group_colours", sns.color_palette("Set2", len(group_names)))

    # Vars without RE
    noRE_vars = [key for key in titles if "RE" not in key]

    # Demographics, anthropometrics and physiological variables ignoring Sex
    stat_comparison, normfigs = compare_0D_contvar_indgroups_one_condition(
        {var: datasheet[var] for var in noRE_vars if "Sex" != var},
        datasheet[grouping_var].values,
        colours=group_colours,
        group_names=group_names,
    )

    group_labels = np.unique(datasheet[grouping_var].values)

    # Compare sex
    fempctge = []
    sextable = []

    for gri, group in enumerate(group_labels):
        # Get pts in that cluster
        groupmaster = datasheet.loc[datasheet[grouping_var] == group]
        fempctge.append(
            len(groupmaster.loc[groupmaster["Sex"] == "Female"])
            / len(groupmaster)
            * 100
        )

        # Get number of women and men in that cluster
        sextable.append(
            [
                len(groupmaster.loc[groupmaster["Sex"] == "Female"]),
                len(groupmaster.loc[groupmaster["Sex"] == "Male"]),
            ]
        )

    # Add chi square test
    stat_comparison['Sex'] = {}
    stat_comparison['Sex']["chi_test"] = {}
    (
        stat_comparison['Sex']["chi_test"]["chi_sq"],
        stat_comparison['Sex']["chi_test"]["p"],
        _,
        _,
    ) = stats.chi2_contingency(sextable)

    # Make figures
    if len(noRE_vars) == 16:
        demoanthrophysfig, axs = plt.subplots(4, 4, figsize=(11, 8))

    elif len(noRE_vars) == 15:
        demoanthrophysfig, axs = plt.subplots(3, 5, figsize=(11, 6))

    else:
        print(
            "Number of variables without RE is not 15 or 16."
            "Figure may look a mess."
            "Please modify the code accordingly."
        )
        nrows = int(np.ceil(len(noRE_vars) / 4))
        demoanthrophysfig, axs = plt.subplots(nrows, 4, figsize=(11, 3 * nrows))

    # Flatten axes
    axs = axs.flatten()

    # Vis variables
    for vari, varname in enumerate(noRE_vars):

        if varname == "Sex":

            # Bar plot
            sns.barplot(
                ax=axs[vari],
                x=group_labels,
                y=fempctge,
                hue=group_labels,
                palette=group_colours,
                legend=False,
            )

            # Set xticks
            axs[vari].set_xticks(axs[vari].get_xticks(), group_names)

        elif varname == "RunningDaysAWeek":
            # Count plot
            sns.countplot(
                ax=axs[vari],
                x=datasheet[varname],
                hue=datasheet[grouping_var],
                palette=group_colours,
                legend=False,
            )

        else:
            # Violin plot
            sns.violinplot(
                ax=axs[vari],
                x=datasheet[grouping_var],
                y=datasheet[varname],
                hue=datasheet[grouping_var],
                palette=group_colours,
                legend=False,
            )

            # Xticks
            axs[vari].set_xticks(axs[vari].get_xticks(), group_names)

        # Yticks for Time10Ks
        if varname == "Time10Ks" or varname == "Sess2_times":
            # Convert to datetime and keep just mm:ss
            yticks = [
                str(timedelta(seconds=x)) for x in axs[vari].get_yticks()
            ]
            yticks = [x[x.find(":") + 1:] for x in yticks]

            # Set new ticks
            axs[vari].set_yticklabels(yticks)

        # Ylabels
        axs[vari].set_ylabel(ylabels[varname])

        # Xlabel off
        axs[vari].set_xlabel("")

        # Title
        if varname in stat_comparison.keys():
            # Get key which is not normality or 'homoscedasticity'
            stat_test = [
                key for key in stat_comparison[varname].keys() if key not in ["normality", "homoscedasticity"]
            ][0]

            # Add asterisk to indicate significant differences
            if stat_comparison[varname][stat_test]["p"] < 0.05:
                axs[vari].set_title(f"{titles[varname]} *")
            else:
                axs[vari].set_title(titles[varname])

        else:
            axs[vari].set_title(titles[varname])

    plt.tight_layout()

    # RE variables

    # Get EE data into a dataframe FIX PT AND SPEEDS
    redf = pd.DataFrame()
    redf["EE"] = np.concatenate(
        [datasheet[f"EE{speed}kg"].values for speed in re_speeds]
    )
    redf["speed"] = np.concatenate(
        [[int(speed)] * len(datasheet.index) for speed in re_speeds]
    )
    redf["clustlabel"] = np.tile(datasheet[grouping_var].values, len(re_speeds))
    redf["ptcode"] = np.tile(datasheet.index, len(re_speeds))

    # Run 2 way ANOVA with one RM factor (speed) and one between factor (cluster)
    stat_comparison["RE"] = anova2onerm_0d_and_posthocs(
        redf, dv="EE", within="speed", between=grouping_var, subject="ptcode"
    )

    refig, reaxs = plt.subplots(1, 3, figsize=(11, 3))
    reaxs = reaxs.flatten()

    # Plot results
    for speedi, speed in enumerate(re_speeds):

        # Violin plot
        sns.violinplot(ax=reaxs[speedi],
                       x=grouping_var,
                       y='EE',
                       data=redf.loc[redf['speed'] == speed],
                       palette=group_colours,
                       hue=grouping_var,
                       legend=False)

        # Add C at the start of each xtick
        reaxs[speedi].set_xticks(reaxs[speedi].get_xticks(),
                                 [f'C{int(x)}' for x in reaxs[speedi].get_xticks()])

        # Add stats in xlabel
        if (stat_comparison['RE']['ANOVA2onerm']['p-unc'].loc[
            stat_comparison['RE']['ANOVA2onerm']['Source'] == 'clustlabel'].values
                < 0.05):

            statsstr = write_0Dposthoc_statstr(stat_comparison['RE']['posthocs'],
                                               f'speed * {grouping_var}',
                                               'speed',
                                               speed)
            reaxs[speedi].set_xlabel(f'C: {statsstr}', fontsize=10)

        else:
            reaxs[speedi].set_xlabel(' ', fontsize=10)

        # y label
        if speedi == 0:
            reaxs[speedi].set_ylabel(ylabels['RE'])
        else:
            reaxs[speedi].set_ylabel('')

        # Add title
        reaxs[speedi].set_title(f'{speed} km/h')

    # Same y limits
    ylims = [ax.get_ylim() for ax in reaxs]
    for ax in reaxs:
        ax.set_ylim([min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])])

    # Set suptitle
    statsstr = write_0DmixedANOVA_statstr(stat_comparison['RE']['ANOVA2onerm'],
                                          between=grouping_var,
                                          within='speed',
                                          betweenlabel='C',
                                          withinlabel='S')

    # Set title
    refig.suptitle(f'Running economy\n{statsstr}')

    plt.tight_layout()

    return stat_comparison, demoanthrophysfig, normfigs, refig


def run_0D_ANOVA2onerm(
        datadict,
        designfactors,
        between_factor,
        within_factor,
        **kwargs
):
    """ """

    # Get kwargs
    titles = kwargs.get("titles", {var: var for var in datadict})
    ylabels = kwargs.get("ylabels", {var: "" for var in datadict})
    group_names = kwargs.get("group_names", [str(x) for x in np.unique(designfactors['group'])])
    group_colours = kwargs.get("group_colours", sns.color_palette("Set2", len(group_names)))
    rm_names = kwargs.get("rm_names", np.unique(designfactors['rm']))
    rm_colours = kwargs.get("rm_colours", sns.color_palette("Set2", len(rm_names)))
    within_label = kwargs.get("within_label", within_factor[0].upper())
    between_label = kwargs.get("between_label", between_factor[0].upper())
    within_vis = kwargs.get("within_vis", True)

    figdict = {}
    stat_comparison = {}
    write_within = True
    if within_vis:
        write_within = False
        figdict["discvars_within_effect"], withinaxs = plt.subplots(
            1, len(datadict.keys()), figsize=(11, 4)
        )
        withinaxs = withinaxs.flatten()

    for vari, varname in enumerate(datadict):

        # Get data
        df = pd.DataFrame(
            {
                varname: datadict[varname],
                within_factor: designfactors["rm"],
                between_factor: designfactors["group"],
                "ptcode": designfactors["ptids"],
            }
        )

        # Run 2 way ANOVA with one RM factor (segment) and one between factor (cluster)
        stat_comparison[varname] = anova2onerm_0d_and_posthocs(
            df,
            dv=varname,
            within=within_factor,
            between=between_factor,
            subject="ptcode",
        )

        # Plot results
        figdict[f"{varname}_ANOVA2onerm"] = visualise_0D_ANOVA2onerm(
            df,
            stat_comparison[varname],
            rm_names=rm_names,
            title=titles[varname],
            ylabel=ylabels[varname],
            group_names=group_names,
            group_colours=group_colours,
            within_factor=within_factor,
            between_factor=between_factor,
            within_label=within_label,
            between_label=between_label,
            write_within=write_within,
        )

        if within_vis:
            _ = plot_0D_ANOVA2onerm_within_effect(
                df,
                stat_comparison[varname],
                ax=withinaxs[vari],
                title=titles[varname],
                ylabel=ylabels[varname],
                rm_names=rm_names,
                rm_colours=rm_colours,
            )

    if within_vis:
        figdict["discvars_within_effect"].tight_layout()

    return figdict, stat_comparison
