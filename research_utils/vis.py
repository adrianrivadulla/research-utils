# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from research_utils.statistics import (
    write_0Dposthoc_statstr,
    write_0DmixedANOVA_statstr,
)


# %% Functions


def visualise_0D_ANOVA2onerm(
    datadf,
    stat_comp,
    **kwargs
):
    """ """

    # Get kwargs
    title = kwargs.get("title", datadf.columns[0])
    ylabel = kwargs.get("ylabel", "")
    within_factor = kwargs.get("within_factor", datadf.columns[1])
    between_factor = kwargs.get("between_factor", datadf.columns[2])
    within_label = kwargs.get("within_label", within_factor[0].upper())
    between_label = kwargs.get("between_label", between_factor[0].upper())
    rm_names = kwargs.get("rm_names", np.unique(datadf[within_factor].values))
    group_names = kwargs.get("group_names", [str(x) for x in datadf[between_factor].unique()])
    group_colours = kwargs.get("group_colours", sns.color_palette("Set2", n_colors=len(group_names)))
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


def plot_0D_ANOVA2onerm_within_effect(
    datadf,
    stat_comparison,
    **kwargs
):
    """ """

    # Get kwargs
    ax = kwargs.get("ax", plt.gca())
    title = kwargs.get("title", datadf.columns[0])
    ylabel = kwargs.get("ylabel", "")
    within_factor = kwargs.get("within_factor", datadf.columns[1])
    rm_names = kwargs.get("rm_names", np.unique(datadf[within_factor].values))
    rm_colours = kwargs.get("rm_colours", sns.color_palette("Set2", n_colors=len(rm_names)))


    # Create segment figure
    sns.violinplot(x=within_factor, y=datadf.columns[0], data=datadf, palette=rm_colours, ax=ax)

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
