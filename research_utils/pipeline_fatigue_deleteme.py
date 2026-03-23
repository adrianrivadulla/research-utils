# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from research_utils.statistics import (
    write_0Dposthoc_statstr,
    write_0DmixedANOVA_statstr,
    write_spm_stats_str,
)
import seaborn as sns
import spm1d


# %% Functions


def visualise_0D_ANOVA2onerm(datadf, stat_comp, **kwargs):
    """ """

    # Get kwargs
    title = kwargs.get("title", datadf.columns[0])
    ylabel = kwargs.get("ylabel", "")
    within_factor = kwargs.get("within_factor", datadf.columns[1])
    between_factor = kwargs.get("between_factor", datadf.columns[2])
    within_label = kwargs.get("within_label", within_factor[0].upper())
    between_label = kwargs.get("between_label", between_factor[0].upper())
    rm_names = kwargs.get("rm_names", np.unique(datadf[within_factor].values))
    group_names = kwargs.get(
        "group_names", [str(x) for x in datadf[between_factor].unique()]
    )
    group_colours = kwargs.get(
        "group_colours", sns.color_palette("Set2", n_colors=len(group_names))
    )
    write_within = kwargs.get("write_within", True)
    write_between = kwargs.get("write_between", True)
    write_interaction = kwargs.get("write_interaction", True)

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs = axs.flatten()

    for rmi, rm in enumerate(rm_names):
        # Violin plot
        sns.violinplot(
            ax=axs[rmi],
            x=between_factor,
            y=datadf[datadf.columns[0]],
            hue=between_factor,
            data=datadf.loc[datadf[within_factor] == rm],
            palette=group_colours,
            legend=False,
        )
        # Xticks
        if group_names != "":
            axs[rmi].set_xticks(range(len(group_names)), group_names)

        # Add stats in xlabel
        if (
            stat_comp["ANOVA2onerm"]["p-unc"]
            .loc[stat_comp["ANOVA2onerm"]["Source"] == between_factor]
            .values
            < 0.05
        ):
            statsstr = write_0Dposthoc_statstr(
                stat_comp["posthocs"],
                f"{within_factor} * {between_factor}",
                within_factor,
                rm,
            )
            axs[rmi].set_xlabel(f"{between_label}: {statsstr}", fontsize=10)

        else:
            axs[rmi].set_xlabel(" ", fontsize=10)

        # y label
        axs[rmi].set_ylabel("")

        # Add title
        axs[rmi].set_title(rm)

    # Same y limits for every plot
    ylims = [ax.get_ylim() for ax in axs]
    for ax in axs:
        ax.set_ylim(
            [min([ylim[0] for ylim in ylims]), max([ylim[1] for ylim in ylims])]
        )

    # Write stats string
    statsstr = write_0DmixedANOVA_statstr(
        stat_comp["ANOVA2onerm"],
        between=between_factor,
        within=within_factor,
        betweenlabel=between_label,
        withinlabel=within_label,
        write_between=write_between,
        write_within=write_within,
        write_interaction=write_interaction,
    )
    # Set title
    fig.suptitle(f"{title}\n{statsstr}")

    # Set ylabel for the first plot only
    axs[0].set_ylabel(ylabel)

    plt.tight_layout()

    return fig


def plot_0D_ANOVA2onerm_within_effect(datadf, stat_comparison, **kwargs):
    """ """

    # Get kwargs
    ax = kwargs.get("ax", plt.gca())
    title = kwargs.get("title", datadf.columns[0])
    ylabel = kwargs.get("ylabel", "")
    within_factor = kwargs.get("within_factor", datadf.columns[1])
    rm_names = kwargs.get("rm_names", np.unique(datadf[within_factor].values))
    rm_colours = kwargs.get(
        "rm_colours", sns.color_palette("Set2", n_colors=len(rm_names))
    )

    # Create segment figure
    sns.violinplot(
        x=within_factor,
        y=datadf.columns[0],
        hue=within_factor,
        data=datadf,
        palette=rm_colours,
        legend=False,
        ax=ax,
    )

    # xlabeloff
    ax.set_xlabel("")

    # Make ylims 20% bigger, 7% on each side
    ylim = ax.get_ylim()
    ax.set_ylim(
        [ylim[0] - (ylim[1] - ylim[0]) * 0.20, ylim[1] + (ylim[1] - ylim[0]) * 0.07]
    )

    # Annotate graph with posthoc stats if significant effect of segment is found
    if (
        stat_comparison["ANOVA2onerm"]["p-unc"]
        .loc[stat_comparison["ANOVA2onerm"]["Source"] == "segment"]
        .values
        < 0.05
    ):
        # Get xticks and xticklabels
        xticks = ax.get_xticks()

        # Go through all within_factor contrasts
        for _, contrast in (
            stat_comparison["posthocs"]
            .loc[stat_comparison["posthocs"]["Contrast"] == within_factor]
            .iterrows()
        ):
            # Get the x position as the mean of the xticks
            x = np.mean(
                [
                    xticks[rm_names.index(contrast["A"])],
                    xticks[rm_names.index(contrast["B"])],
                ]
            )

            # Annotate stats if mid, put it at the bottom and two lines
            if contrast["A"] == rm_names[1] or contrast["B"] == rm_names[1]:
                y = ylim[0]
                if contrast["p-corr"] < 0.001:
                    strstats = (
                        f"t = {np.round(contrast['T'], 2)}, p < 0.001,\n"
                        f"d = {np.round(contrast['cohen'], 2)}[{np.round(contrast['esci95_low'], 2)}, {np.round(contrast['esci95_up'], 2)}]"
                    )
                else:
                    strstats = (
                        f"t = {np.round(contrast['T'], 2)}, p = {np.round(contrast['p-corr'], 3)},\n"
                        f"d = {np.round(contrast['cohen'], 2)}[{np.round(contrast['esci95_low'], 2)}, {np.round(contrast['esci95_up'], 2)}]"
                    )

                # Annotate
                ax.annotate(
                    strstats, (x, y - y * 0.02), ha="center", va="top", fontsize=10
                )

            # if end and start, put it at the top in one single line
            else:
                y = ylim[1]
                if contrast["p-corr"] < 0.001:
                    strstats = f"t = {np.round(contrast['T'], 2)}, p < 0.001, d = {np.round(contrast['cohen'], 2)}[{np.round(contrast['esci95_low'], 2)}, {np.round(contrast['esci95_up'], 2)}]"
                else:
                    strstats = f"t = {np.round(contrast['T'], 2)}, p = {np.round(contrast['p-corr'], 3)}, d = {np.round(contrast['cohen'], 2)}[{np.round(contrast['esci95_low'], 2)}, {np.round(contrast['esci95_up'], 2)}]"

                # Annotate
                ax.annotate(strstats, (x, y), ha="center", va="bottom", fontsize=10)

            # Hline to indicate the significant difference in post hoc test
            if (
                contrast["A"] == rm_names[1]
                and contrast["B"] == rm_names[2]
                or contrast["B"] == rm_names[1]
                and contrast["A"] == rm_names[2]
            ):
                x1 = xticks[rm_names.index(rm_names[1])] + 0.05
                x2 = xticks[rm_names.index(rm_names[2])] - 0.05

            elif (
                contrast["A"] == rm_names[1]
                and contrast["B"] == rm_names[0]
                or contrast["B"] == rm_names[1]
                and contrast["A"] == rm_names[0]
            ):
                x1 = xticks[rm_names.index(rm_names[1])] - 0.05
                x2 = xticks[rm_names.index(rm_names[0])] + 0.05

            else:
                x1 = xticks[rm_names.index(rm_names[0])] + 0.05
                x2 = xticks[rm_names.index(rm_names[2])] - 0.05

            # Draw line
            ax.hlines(y, x1, x2, color="k", linewidth=0.5)

    # Set ylabel
    ax.set_ylabel(ylabel)

    if (
        stat_comparison["ANOVA2onerm"]["p-unc"]
        .loc[stat_comparison["ANOVA2onerm"]["Source"] == within_factor]
        .values
        < 0.001
    ):
        strstats = (
            f"F = {np.round(stat_comparison['ANOVA2onerm']['F'].values[1], 2)},"
            f" p < 0.001"
        )
    else:
        strstats = (
            f"F = {np.round(stat_comparison['ANOVA2onerm']['F'].values[1], 2)},"
            f" p = {np.round(stat_comparison['ANOVA2onerm']['p-unc'].values[1], 3)}"
        )

    # Set title
    ax.set_title(f"{title} ({strstats})")

    return ax


def vis_SPM_ANOVA2onerm_between_and_x_effects(datadict, designfactors, stat_comparison, **kwargs):
    """ """

    # Get kwargs
    rm_names = kwargs.get("rm_names", np.unique(designfactors["rm"]))
    suptitles = kwargs.get("suptitles", {key: key for key in datadict.keys()})
    ylabels = kwargs.get("ylabels", {key: "" for key in datadict.keys()})
    xlabels = kwargs.get("xlabels", {key: "Time (%)" for key in datadict.keys()})
    group_names = kwargs.get("group_names", np.unique(designfactors["group"]))
    colours = kwargs.get("colours", sns.color_palette("Set2", n_colors=len(group_names)))
    between_label = kwargs.get("between_label", "B")
    within_label = kwargs.get("within_label", "W")
    vline_var = kwargs.get("vline_var", None)

    figs = {}

    for vari, var in enumerate(datadict.keys()):

        # Prepare data for SPM mean and std plots
        figs[var] = plt.figure()
        figs[var].set_size_inches(10, 5)
        basegrid = figs[var].add_gridspec(2, 1)
        topgrid = basegrid[0].subgridspec(1, len(np.unique(designfactors["rm"])))
        bottomgrid = basegrid[1].subgridspec(1, len(np.unique(designfactors["rm"])) - 1)

        upperaxs = []
        loweraxs = []

        # Extract relevant stats from stat_comparison for current variable
        for rmfi, rmfactor in enumerate(rm_names):

            # Create axis in group and interaction figure
            upperaxs.append(figs[var].add_subplot(topgrid[0, rmfi]))

            # Plot mean and std curves
            for group in np.unique(designfactors["group"]):
                idcs = np.where((designfactors["group"] == group) & (designfactors["rm"] == rmfactor))[0]

                spm1d.plot.plot_mean_sd(
                    datadict[var][idcs, :],
                    x=np.linspace(0, 100, datadict[var].shape[1]),
                    ax=upperaxs[rmfi],
                    linecolor=colours[group],
                    facecolor=colours[group],
                )

            # Add vertical line (at avge toe off) for each group (outside loop so it doesn't mess the ylims)
            if vline_var is not None:
                for group in np.unique(designfactors["group"]):
                    idcs = np.where((designfactors["group"] == group) & (designfactors["rm"] == rmfactor))[0]

                    upperaxs[rmfi].axvline(x=np.mean(vline_var[idcs]) * 100, color=colours[group], linestyle=":")

            # xlabel
            upperaxs[rmfi].set_xlabel(xlabels[var], fontsize=10)

            # Title
            upperaxs[rmfi].set_title(rmfactor)

            # Add patches to upperaxs if significant diffs were found
            if stat_comparison[var]["ANOVA2onerm"][0].h0reject:
                between_posthocs = stat_comparison[var]["posthocs"]["group"]
                if between_posthocs[rmfactor]["snpm_ttest2"].h0reject:
                    # Scaler for sigcluster endpoints
                    tscaler = upperaxs[rmfi].get_xlim()[1] / (datadict[var].shape[1] - 1)

                    # Add significant pathces to upperaxs
                    add_sig_spm_cluster_patch(
                        upperaxs[rmfi], between_posthocs[rmfactor]["snpm_ttest2"], tscaler=tscaler
                    )

                    # Add stats to title
                    curr_title = upperaxs[rmfi].get_title()
                    statstr = f"t* = {write_spm_stats_str(between_posthocs[rmfactor]['snpm_ttest2'], mode='full')}"
                    upperaxs[rmfi].set_title(f"{curr_title}\n{statstr}", fontsize=10)

            if rmfi > 0:
                # Plot change in variable by group
                loweraxs.append(figs[var].add_subplot(bottomgrid[0, rmfi - 1]))

                # Add horizontal line at 0
                loweraxs[-1].axhline(0, color="black", linestyle="-", linewidth=0.5, zorder=1)

                for group in np.unique(designfactors["group"]):
                    # Get indices of group at current measure
                    idcs = np.where((designfactors["group"] == group) & (designfactors["rm"] == rmfactor))[0]

                    # Get indices of group at previous repeated measure
                    idcsprev = np.where(
                        (designfactors["rm"] == rm_names[rmfi - 1]) & (designfactors["group"] == group)
                    )[0]

                    # Calculate difference in variable between groups
                    Ydiff = datadict[var][idcs, :] - datadict[var][idcsprev, :]

                    spm1d.plot.plot_mean_sd(
                        Ydiff,
                        x=np.linspace(0, 100, Ydiff.shape[1]),
                        linecolor=colours[group],
                        facecolor=colours[group],
                        ax=loweraxs[-1],
                    )

                # Add vertical lines (at avge toe off) for each group (outside loop so it doesn't mess the ylims)
                if vline_var is not None:
                    for group in np.unique(designfactors["group"]):
                        gridcs = np.where((designfactors["rm"] == rmfactor) & (designfactors["group"] == group))[0]

                        loweraxs[-1].axvline(x=np.mean(vline_var[gridcs]) * 100, color=colours[group], linestyle=":")

                # Title
                loweraxs[-1].set_title(f"{rmfactor} with respect to {rm_names[rmfi - 1]}")

                # xlabel
                loweraxs[-1].set_xlabel(xlabels[var], fontsize=10)

                # Add patches to loweraxs if significant diffs are found
                if stat_comparison[var]["ANOVA2onerm"][2].h0reject:
                    interaction_posthocs = stat_comparison[var]["posthocs"]["interaction"]
                    if interaction_posthocs[f"{rm_names[rmfi]}_wrt_{rm_names[rmfi - 1]}"]["snpm_ttest2"].h0reject:
                        # Scaler for sigcluster endpoints
                        tscaler = loweraxs[rmfi].get_xlim()[1] / (Ydiff.shape[1] - 1)

                        # Add significant pathces to upperaxs
                        add_sig_spm_cluster_patch(
                            loweraxs[rmfi],
                            interaction_posthocs[f"{rm_names[rmfi]}_wrt_{rm_names[rmfi - 1]}"]["snpm_ttest2"],
                            tscaler=tscaler,
                        )

                        # Add stats to xlabel
                        statstr = f"t* = {write_spm_stats_str(interaction_posthocs[f'{rm_names[rmfi]}_wrt_{rm_names[rmfi - 1]}']['snpm_ttest2'], mode='full')}"
                        loweraxs[rmfi].set_xlabel(statstr, fontsize=10)

        # Legend
        loweraxs[-1].legend(
            ["_nolegend_"] + group_names,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=2,
            bbox_transform=figs[var].transFigure,
            frameon=False,
        )
        plt.subplots_adjust(bottom=0.11)

        # ylabels
        upperaxs[0].set_ylabel(ylabels[var])

        # Get units and add them to ylabels of loweraxs
        unitstr = ylabels[var].split("(")[1].split(")")[0]
        loweraxs[0].set_ylabel(f"$\Delta$ ({unitstr})")

        # Get ylims for all loweraxs
        ylims = [ax.get_ylim() for ax in upperaxs]

        # Set ylims for all loweraxs
        for ax in upperaxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # Get ylims for all loweraxs
        ylims = [ax.get_ylim() for ax in loweraxs]

        # Set ylims for all loweraxs
        for ax in loweraxs:
            ax.set_ylim([min([x[0] for x in ylims]), max([x[1] for x in ylims])])

        # Supttitle
        # Write between effect string for suptitle
        statstr = f"{between_label}: F* = {write_spm_stats_str(stat_comparison[var]['ANOVA2onerm'][0], mode='full')}"

        # Write interaction effect string for suptitle
        statstr += f"; {between_label}x{within_label}: F* = {write_spm_stats_str(stat_comparison[var]['ANOVA2onerm'][2], mode='full')}"

        figs[var].suptitle(f"{suptitles[var]}\n{statstr}")
        figs[var].tight_layout()

    return figs


def vis_SPM_ANOVA2onerm_within_effect(datadict, designfactors, stat_comparison, **kwargs):
    """ """

    # Get kwargs
    rm_names = kwargs.get("rm_names", np.unique(designfactors["rm"]))
    fig_rows = kwargs.get("fig_rows", 1)
    fig_cols = kwargs.get("fig_cols", len(datadict))
    colours = kwargs.get("colours", sns.color_palette("Set2", n_colors=len(rm_names)))
    titles = kwargs.get("titles", {key: key for key in datadict.keys()})
    ylabels = kwargs.get("ylabels", {key: "" for key in datadict.keys()})
    xlabels = kwargs.get("xlabels", {key: "Time (%)" for key in datadict.keys()})
    vline_var = kwargs.get("vline_var", None)

    # Repeated measures
    rmffig, rmfaxs = plt.subplots(fig_rows, fig_cols, figsize=(11, 4.5))
    rmfaxs = rmfaxs.flatten()

    for vari, var in enumerate(datadict.keys()):
        for rmfi, rmfactor in enumerate(rm_names):
            # Repeated measures figure
            rmfidcs = np.where(designfactors["rm"] == rmfactor)[0]
            spm1d.plot.plot_mean_sd(
                datadict[var][rmfidcs, :],
                x=np.linspace(0, 100, datadict[var].shape[1]),
                linecolor=colours[rmfi],
                facecolor=colours[rmfi],
                ax=rmfaxs[vari],
            )

            # x and y labels
            rmfaxs[vari].set_xlabel(xlabels[var], fontsize=10)
            rmfaxs[vari].set_ylabel(ylabels[var])

        # Add vertical line to rm figures (at avge toe off, outside loop so it doesn't mess the ylims)
        if vline_var is not None:
            for rmfi, rmfactor in enumerate(rm_names):
                rmfaxs[vari].axvline(
                    x=np.mean(vline_var[designfactors["rm"] == rmfactor]) * 100, color=colours[rmfi], linestyle=":"
                )

        # Add title to with within ANOVA effect in the title
        statsstr = f"F* = {np.round(stat_comparison[var]['ANOVA2onerm'][1].zstar, 2)}"
        rmfaxs[vari].set_title(f"{titles[var]}\n{statsstr}")

        # Add patches to if significant diffs are found
        for comparison in stat_comparison[var]["posthocs"]["rm"].values():
            if comparison["snpm_ttest2"].h0reject:
                # Scaler for sigcluster endpoints
                tscaler = rmfaxs[vari].get_xlim()[1] / (datadict[var].shape[1] - 1)

                for sigcluster in comparison["snpm_ttest2"].clusters:
                    ylim = rmfaxs[vari].get_ylim()
                    rmfaxs[vari].add_patch(
                        plt.Rectangle(
                            (sigcluster.endpoints[0] * tscaler, ylim[0]),
                            (sigcluster.endpoints[1] - sigcluster.endpoints[0]) * tscaler,
                            ylim[1] - ylim[0],
                            color="grey",
                            alpha=0.5,
                            linestyle="",
                        )
                    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    rmfaxs[-2].legend(
        rm_names, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=rmffig.transFigure, frameon=False
    )

    return rmffig


def add_sig_spm_cluster_patch(ax, spmobj, tscaler=1):
    """
    Add patches to a plot to indicate significant clusters from SPM (Statistical Parametric Mapping) analysis.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to which the patches will be added.
    spmobj (object): The SPM object containing the significant clusters.
    tscaler (float, optional): A scaling factor for the time axis. Defaults to 1.
    """

    for sigcluster in spmobj.clusters:
        ax.axvspan(
            sigcluster.endpoints[0] * tscaler, sigcluster.endpoints[1] * tscaler, color="grey", alpha=0.5, linestyle=""
        )
