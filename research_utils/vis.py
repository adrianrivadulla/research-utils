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
    varname,
    rmlabels,
    colours,
    title,
    ylabel,
    group_names="",
    within_factor="",
    between_factor="",
    within_label="",
    between_label="",
    write_between=True,
    write_within=True,
    write_interaction=True,
):
    """ """

    if within_factor == "":
        within_factor = "within"
    if between_factor == "":
        between_factor = "between"
    if within_label == "":
        within_label = "W"
    if between_label == "":
        between_label = "B"

    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs = axs.flatten()

    for rmi, rm in enumerate(rmlabels):
        # Violin plot
        sns.violinplot(
            ax=axs[rmi],
            x=between_factor,
            y=varname,
            data=datadf.loc[datadf[within_factor] == rm],
            palette=colours,
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
            axs[rmi].set_xlabel(f"C: {statsstr}", fontsize=10)

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
    varname,
    rmlabels,
    rm_colours,
    title,
    ylabel,
    ax,
    within_factor,
):
    """ """

    # TODO.  Check here there are direct references to segments that need to be replaced with the rmlabels

    # Create segment figure
    sns.violinplot(x=within_factor, y=varname, data=datadf, palette=rm_colours, ax=ax)

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
                    xticks[rmlabels.index(contrast["A"])],
                    xticks[rmlabels.index(contrast["B"])],
                ]
            )

            # Annotate stats if mid, put it at the bottom and two lines
            if contrast["A"] == "mid" or contrast["B"] == "mid":
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
                contrast["A"] == "mid"
                and contrast["B"] == "end"
                or contrast["B"] == "mid"
                and contrast["A"] == "end"
            ):
                x1 = xticks[rmlabels.index("mid")] + 0.05
                x2 = xticks[rmlabels.index("end")] - 0.05

            elif (
                contrast["A"] == "mid"
                and contrast["B"] == "start"
                or contrast["B"] == "mid"
                and contrast["A"] == "start"
            ):
                x1 = xticks[rmlabels.index("mid")] - 0.05
                x2 = xticks[rmlabels.index("start")] + 0.05

            else:
                x1 = xticks[rmlabels.index("start")] + 0.05
                x2 = xticks[rmlabels.index("end")] - 0.05

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
