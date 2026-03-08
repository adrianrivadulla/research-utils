"""
TODO.

Bring single_speed_kinematics_comparison (in clustering repo) and run_single_condition_stats (in dimred repo) together because they do pretty much the same

Pipeline_fatigue and pipeline_clustering are the same for 0D, multicondition (segments and speeds respectively) stats, so can also be merged together.
The only difference is the plotting, which can be done with if statements.

Then the second part of multispeed_kinematics_comparison is the same as run_SPM_ANOVA2onerm.
You need to rearrange the data in the clustering script to be in the same format as the fatigue script,
which is more aligned with SPM, and then you can run the same SPM code for both.

"""

# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_utils.statistics import anova2onerm_0d_and_posthocs, write_0Dposthoc_statstr, write_0DmixedANOVA_statstr
from research_utils.vis import visualise_0D_ANOVA2onerm, plot_0D_ANOVA2onerm_within_effect
import seaborn as sns

#%% Functions

def run_0D_ANOVA2onerm(datadict, designfactors, titles, ylabels, group_colours, rmlabels, group_names, between_factor, within_factor, between_label, within_label, within_vis=True, rm_colours=None):
    """

    """

    figdict = {}
    stat_comparison = {}

    if within_vis:
        write_within = False
        figdict['discvars_within_effect'], withinaxs = plt.subplots(1, len(datadict.keys()), figsize=(11, 4))
        withinaxs = withinaxs.flatten()

        if rm_colours is None:
            rm_colours = sns.color_palette('Set2', len(rmlabels))

    for vari, varname in enumerate(datadict.keys()):

        # Get data
        df = pd.DataFrame()
        df[varname] = datadict[varname]
        df[within_factor] = designfactors['rm']
        df[between_factor] = designfactors['group']
        df['ptcode'] = designfactors['ptids']

        # Run 2 way ANOVA with one RM factor (segment) and one between factor (cluster)
        stat_comparison[varname] = anova2onerm_0d_and_posthocs(df,
                                                               dv=varname,
                                                               within=within_factor,
                                                               between=between_factor,
                                                               subject='ptcode')

        # Plot results
        figdict[f'{varname}_ANOVA2onerm'] = visualise_0D_ANOVA2onerm(df,
                                                                     stat_comparison[varname],
                                                                     varname,
                                                                     rmlabels,
                                                                     group_colours,
                                                                     titles[varname],
                                                                     ylabels[varname],
                                                                     group_names=group_names,
                                                                     within_factor=within_factor,
                                                                     between_factor=between_factor,
                                                                     within_label=within_label,
                                                                     between_label=between_label,
                                                                     write_within=write_within,
                                                                     )

        if within_vis:

            _ = plot_0D_ANOVA2onerm_within_effect(df,
                                                  stat_comparison[varname],
                                                  varname,
                                                  rmlabels,
                                                  rm_colours,
                                                  titles[varname],
                                                  ylabels[varname],
                                                  withinaxs[vari],
                                                  within_factor)

    if within_vis:
        figdict['discvars_within_effect'].tight_layout()

    return figdict, stat_comparison
