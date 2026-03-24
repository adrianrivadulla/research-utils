from itertools import combinations
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import pingouin as pg
from scikit_posthocs import posthoc_ttest, posthoc_dunn
from scipy import stats
import seaborn as sns
import spm1d
import statsmodels.api as sm


# TODO.

"""

- Refactor to separate stats and figures.
- Env file for all these imports
- Simplify var names in functions
- Use snake_case for all var names in functions

- Move here scripts for demoanthrophys analysis

For stats, you need:

1. single speed/condition test (used in clustering and dim red papers)
2. multicondition test with one RM factor and one between factor (used in clustering and fatigue papers)

Revise namings and move figure saving logic outside of functions

"""
# TODO.


def SPM_ANOVA2onerm(datadict, designfactors, random_seed=None, **kwargs):
    """ """

    # Get kwargs
    rm_names = kwargs.get("rm_names", np.unique(designfactors["rm"]))

    stat_comparison = {}
    figs = {}

    if random_seed is not None:
        np.random.seed(random_seed)

    for vari, var in enumerate(datadict.keys()):
        stat_comparison[var] = {}

        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n\n[{now}] - Conducting ANOVA2onerm for {var}...\n")

        # Replace string labels in segments and pt with integers
        rmcodes = pd.Categorical(
            designfactors["rm"], categories=rm_names, ordered=True
        ).codes
        ptcodes = pd.Categorical(designfactors["ptids"]).codes

        # Conduct SPM analysis
        spmlist = spm1d.stats.nonparam.anova2onerm(
            datadict[var], designfactors["group"], rmcodes, ptcodes
        )

        stat_comparison[var]["ANOVA2onerm"] = spmlist.inference(
            alpha=0.05, iterations=1000
        )

        print(stat_comparison[var]["ANOVA2onerm"])

        # Post hoc tests and figures
        stat_comparison[var]["posthocs"] = {}

        # Follow up with post-hoc tests if group effects are found
        if stat_comparison[var]["ANOVA2onerm"][0].h0reject:
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{now}] - Group effect found for {var}, conducting post-hoc tests...\n"
            )
            stat_comparison[var]["posthocs"]["group"] = {}

            # For each repeated measure
            for rmfi, rmfactor in enumerate(rm_names):
                print(f"RM {rmfactor}\n")
                stat_comparison[var]["posthocs"]["group"][rm_names[rmfi]] = {}

                # Get data
                Y = []
                for group in np.unique(designfactors["group"]):
                    # Get indices of clust at current measure
                    gridcs = np.where(
                        (designfactors["rm"] == rmfactor)
                        & (designfactors["group"] == group)
                    )[0]

                    # Append data to groups
                    Y.append(datadict[var][gridcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Y[0], Y[1])
                snpmi = snpm.inference(
                    alpha=0.05 / len(rm_names), two_tailed=True, iterations=1000
                )
                print(snpmi)

                # Add snpmi to dictionary
                stat_comparison[var]["posthocs"]["group"][rmfactor]["snpm_ttest2"] = (
                    snpmi
                )

                # SPM figure for current posthoc test
                figs[f"{var}_posthoc_group_at_{rmfactor}"] = plot_spm_test(
                    snpmi, f"{var}_posthoc_group_at_{rmfactor}"
                )

        # RM factor effect
        if stat_comparison[var]["ANOVA2onerm"][1].h0reject:
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{now}] - Within effect found for {var}, conducting post-hoc tests...\n"
            )
            stat_comparison[var]["posthocs"]["rm"] = {}

            # Get all possible combinations of segments
            rmcombos = list(combinations(range(len(rm_names)), 2))

            # Calculate change in conditions
            for rmcombo in rmcombos:
                print(f"RM {rm_names[rmcombo[0]]} v {rm_names[rmcombo[1]]}\n")

                # Get data
                Y = []
                for rmf in rmcombo:
                    # Get indices of clust at current segment
                    rmfidcs = np.where(rmcodes == rmf)[0]

                    # Append data to Y
                    Y.append(datadict[var][rmfidcs, :])

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Y[0], Y[1])
                snpmi = snpm.inference(
                    alpha=0.05 / len(rmcombos), two_tailed=True, iterations=1000
                )
                print(snpmi)

                # Add snpmi to dictionary
                stat_comparison[var]["posthocs"]["rm"][
                    f"{rm_names[rmcombo[0]]}_v_{rm_names[rmcombo[1]]}"
                ] = {}
                stat_comparison[var]["posthocs"]["rm"][
                    f"{rm_names[rmcombo[0]]}_v_{rm_names[rmcombo[1]]}"
                ]["snpm_ttest2"] = snpmi

                # SPM figure for current posthoc test
                figs[
                    f"{var}_posthoc_rm_{rm_names[rmcombo[0]]}_v_{rm_names[rmcombo[1]]}"
                ] = plot_spm_test(
                    snpmi,
                    f"{var}_posthoc_rm_{rm_names[rmcombo[0]]}_v_{rm_names[rmcombo[1]]}",
                )

        # Interaction effect
        if stat_comparison[var]["ANOVA2onerm"][2].h0reject:
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{now}] - Interaction effect found for {var}, conducting post-hoc tests...\n"
            )
            stat_comparison[var]["posthocs"]["interaction"] = {}

            # Calculate change in conditions
            for rmfi in range(len(rm_names) - 1):
                print(
                    f"{rm_names[rmfi + 1]} with respect to {rm_names[rmfi]} by group\n"
                )

                # Get data
                Ydiff = []
                for group in np.unique(designfactors["group"]):
                    gridcs = np.where(
                        (designfactors["rm"] == rm_names[rmfi])
                        & (designfactors["group"] == group)
                    )[0]
                    gridcsnext = np.where(
                        (designfactors["rm"] == rm_names[rmfi + 1])
                        & (designfactors["group"] == group)
                    )[0]

                    # Append data to groups
                    Ydiff.append(
                        datadict[var][gridcsnext, :] - datadict[var][gridcs, :]
                    )

                # SnPM ttest
                snpm = spm1d.stats.nonparam.ttest2(Ydiff[0], Ydiff[1])
                snpmi = snpm.inference(
                    alpha=0.05 / (len(rm_names) - 1), two_tailed=True, iterations=1000
                )
                print(snpmi)

                # Add snpmi to dictionary
                stat_comparison[var]["posthocs"]["interaction"][
                    f"{rm_names[rmfi + 1]}_wrt_{rm_names[rmfi]}"
                ] = {}
                stat_comparison[var]["posthocs"]["interaction"][
                    f"{rm_names[rmfi + 1]}_wrt_{rm_names[rmfi]}"
                ]["snpm_ttest2"] = snpmi

                # SPM figure for current posthoc test
                figs[f"{var}_posthoc_x_{rm_names[rmfi + 1]}_wrt_{rm_names[rmfi]}"] = (
                    plot_spm_test(
                        snpmi,
                        f"{var}_posthoc_x_{rm_names[rmfi + 1]}_v_{rm_names[rmfi]}",
                    )
                )

    return stat_comparison, figs


def anova2onerm_0d_and_posthocs(datadf, dv="", within="", between="", subject=""):
    """
    Perform a two-way ANOVA with one repeated measures factor and one between-subjects factor,
    followed by Bonferroni post-hoc tests.

    Parameters:
    datadf (pd.DataFrame): The data frame containing the data.
    dv (str): The dependent variable.
    within (str): The within-subjects factor.
    between (str): The between-subjects factor.
    subject (str): The subject identifier.

    Returns:
    statsdict (dict): A dictionary containing the ANOVA results and post-hoc test results.
    """

    statsdict = {}

    # Run ANOVA with one RM factor and one between factor
    statsdict["ANOVA2onerm"] = pg.mixed_anova(
        dv=dv,
        within=within,
        subject=subject,
        between=between,
        data=datadf,
        effsize="np2",
        correction=True,
    )

    # Run Bonferroni post-hoc tests
    statsdict["posthocs"] = pg.pairwise_ttests(
        dv=dv,
        within=within,
        subject=subject,
        between=between,
        data=datadf,
        padjust="bonf",
        effsize="cohen",
    )

    # Add 95% CI to posthocs
    statsdict["posthocs"]["esci95_low"] = np.nan
    statsdict["posthocs"]["esci95_up"] = np.nan
    for i, row in statsdict["posthocs"].iterrows():
        if row["Paired"]:
            ci = pg.compute_esci(
                row["cohen"],
                nx=len(datadf[subject].unique()),
                ny=len(datadf[subject].unique()),
                paired=True,
                eftype="cohen",
                confidence=0.95,
            )
        else:
            subset = datadf.drop_duplicates(subset=[subject], keep="first")

            ci = pg.compute_esci(
                row["cohen"],
                nx=len(subset.loc[subset[between] == row["A"]]),
                ny=len(subset.loc[subset[between] == row["B"]]),
                paired=False,
                eftype="cohen",
                confidence=0.95,
            )

        statsdict["posthocs"].loc[i, "esci95_low"] = ci[0]
        statsdict["posthocs"].loc[i, "esci95_up"] = ci[1]

    return statsdict


def compare_0D_contvar_indgroups_one_condition(datadict, grouping, **kwargs):
    """
    Compare continuous variables between independent groups using various statistical tests.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    grouping (list or np.ndarray): List or array containing the group labels for each data point.
    title_kword (str): Keyword to be used in the title of the plots.
    figdir (str): Directory where the plots will be saved.
    colours (list or np.ndarray): List or array containing the colors for the groups.

    Returns:
    disc_comp (dict): A dictionary containing the results of the statistical tests.
    figs (dict): A dictionary containing the figures generated for normality checks.
    """

    # Get kwargs
    colours = kwargs.get("colours", sns.color_palette("Set2", len(np.unique(grouping))))
    group_names = kwargs.get("group_names", np.unique(grouping))

    disc_comp = {}
    figs = {}

    for key, values in datadict.items():
        disc_comp[key] = {}

        # Check for nans
        if np.any(np.isnan(values)):
            print(f"NaNs found in {key} and they will be removed.")

        # Get variable in groups
        holder = pd.DataFrame({key: np.squeeze(values)})
        holder["grouping"] = grouping
        groups = [
            holder.groupby(["grouping"]).get_group(x)[key].dropna()
            for x in np.sort(holder["grouping"].dropna().unique())
        ]

        # Run normality tests
        disc_comp[key]["normality"] = {}

        figs[key], axes = plt.subplots(1, len(groups))
        figs[key].set_size_inches([11, 3.3])

        # test trigger
        param_route = 1

        for labi, group in enumerate(groups):
            disc_comp[key]["normality"][str(labi)] = {}
            (
                disc_comp[key]["normality"][str(labi)]["W_stat"],
                disc_comp[key]["normality"][str(labi)]["p"],
            ) = stats.shapiro(group)

            # if there were violations of normality or homoscedasticity change trigger for tests later
            if disc_comp[key]["normality"][str(labi)]["p"] <= 0.05:
                param_route = 0

            # Q-Q plots
            sm.qqplot(
                group,
                ax=axes[labi],
                markeredgecolor=colours[labi],
                markerfacecolor=colours[labi],
                line="r",
            )
            # This is so goofy but sm.qqplot doesn't take a line colour argument and I need to change it here
            axes[labi].get_lines()[1].set_color("black")

            # Set labels and title
            axes[labi].set_xlabel(group_names[labi])

            if disc_comp[key]["normality"][str(labi)]["p"] < 0.001:
                axes[labi].set_title(
                    "W: "
                    + str(np.round(disc_comp[key]["normality"][str(labi)]["W_stat"], 3))
                    + "; p < 0.001"
                )
            else:
                axes[labi].set_title(
                    "W: "
                    + str(np.round(disc_comp[key]["normality"][str(labi)]["W_stat"], 3))
                    + "; p = "
                    + str(np.round(disc_comp[key]["normality"][str(labi)]["p"], 3))
                )

        figs[key].suptitle(key)
        plt.tight_layout()

        # Parametric route
        if param_route:
            if len(groups) == 2:
                # Run heteroscedasticity tests
                disc_comp[key]["homoscedasticity"] = {}
                (
                    disc_comp[key]["homoscedasticity"]["Levene_stat"],
                    disc_comp[key]["homoscedasticity"]["p"],
                ) = stats.levene(*groups)

                if disc_comp[key]["homoscedasticity"]["p"] > 0.05:
                    # Independent standard t-test
                    disc_comp[key]["ttest_ind"] = {}
                    (
                        disc_comp[key]["ttest_ind"]["t"],
                        disc_comp[key]["ttest_ind"]["p"],
                    ) = stats.ttest_ind(*groups)

                else:
                    # Welch's t-test
                    disc_comp[key]["ttest_ind"] = {}
                    (
                        disc_comp[key]["ttest_ind"]["welch_t"],
                        disc_comp[key]["ttest_ind"]["p"],
                    ) = stats.ttest_ind(*groups, equal_var=False)

                # Get Cohen's d
                disc_comp[key]["ttest_ind"]["Cohens_d"] = (
                    np.mean(groups[0]) - np.mean(groups[1])
                ) / np.sqrt(
                    (np.std(groups[0], ddof=1) ** 2 + np.std(groups[1], ddof=1) ** 2)
                    / 2
                )

                # Get Hedge's g
                disc_comp[key]["ttest_ind"]["Hedges_g"] = disc_comp[key]["ttest_ind"][
                    "Cohens_d"
                ] * (1 - (3 / (4 * (len(groups[0]) + len(groups[1]) - 2) - 1)))

            elif len(groups) > 2:
                # One-way ANOVA
                disc_comp[key]["ANOVA_1"] = {}
                disc_comp[key]["ANOVA_1"]["F_stat"], disc_comp[key]["ANOVA_1"]["p"] = (
                    stats.f_oneway(*groups)
                )

                if disc_comp[key]["ANOVA_1"]["p"] <= 0.05:
                    # Bonferroni post hoc tests
                    disc_comp[key]["Bonferroni_post_hoc"] = posthoc_ttest(
                        groups, p_adjust="bonferroni"
                    )

        # Non-parametric route
        else:
            if len(groups) == 2:
                # Mann-Whitney U test
                disc_comp[key]["mann_whitney_U"] = {}
                (
                    disc_comp[key]["mann_whitney_U"]["U_stat"],
                    disc_comp[key]["mann_whitney_U"]["p"],
                ) = stats.mannwhitneyu(*groups)

            elif len(groups) > 2:
                # Kruskal
                disc_comp[key]["Kruskal"] = {}
                disc_comp[key]["Kruskal"]["Hstat"], disc_comp[key]["Kruskal"]["p"] = (
                    stats.kruskal(*groups)
                )

                if disc_comp[key]["Kruskal"]["p"] <= 0.05:
                    # Dunn post hoc tests
                    disc_comp[key]["Dunn_post_hoc"] = posthoc_dunn(
                        groups, p_adjust="bonferroni"
                    )

    return disc_comp, figs


def compare_1D_contvar_indgroups_one_condition(
    datadict, grouping, title_kword, figdir, colours
):
    """
    Compare continuous variables between independent groups using SPM1D non-parametric tests.

    Parameters:
    datadict (dict): Dictionary containing the data to be compared.
    grouping (list or np.ndarray): List or array containing the group labels for each data point.
    title_kword (str): Keyword to be used in the title of the plots.
    figdir (str): Directory where the plots will be saved.
    colours (list or np.ndarray): List or array containing the colors for the groups.

    Returns:
    cont_comp (dict): A dictionary containing the results of the statistical tests.
    """

    cont_comp = {}
    varfigs = {}

    for key, values in datadict.items():
        cont_comp[key] = {}

        # Get variable in groups
        groups = [
            values[np.where(grouping == x)[0], :]
            for x in natsort.natsorted(np.unique(grouping))
        ]

        if len(groups) == 2:
            # Non param ttest
            nonparam_ttest2 = spm1d.stats.nonparam.ttest2(groups[0], groups[1])
            cont_comp[key]["np_ttest2"] = nonparam_ttest2.inference(
                alpha=0.05, two_tailed=True, iterations=1000
            )

            # Vis
            varfigs[f"{key}_np_ttest2"], axs = plt.subplots(1, 2, figsize=(10, 4))

            # Average and std patterns by group
            for group, colour in zip(groups, colours):
                spm1d.plot.plot_mean_sd(
                    group, linecolor=colour, facecolor=colour, ax=axs[0]
                )
            axs[0].title(key)

            plot_spm_test(cont_comp[key]["np_ttest2"], title=key, ax=axs[1])
            plt.tight_layout()

        elif len(groups) > 2:
            # Non parametric ANOVA
            nonparam_ANOVA = spm1d.stats.nonparam.anova1(values, grouping)
            cont_comp[key]["np_ANOVA"] = nonparam_ANOVA.inference(
                alpha=0.05, iterations=500
            )

            # Vis
            varfigs[f"{key}_np_ttest2"], axs = plt.subplots(1, 2, figsize=(10, 4))

            # Average and std patterns by group
            for group, colour in zip(groups, colours):
                spm1d.plot.plot_mean_sd(
                    group, linecolor=colour, facecolor=colour, ax=axs[0]
                )
                plt.title(key)

            plot_spm_test(cont_comp[key]["np_ANOVA"], title=key, ax=axs[1])
            plt.tight_layout()

            if cont_comp[key]["np_ANOVA"].h0reject:
                # Adjust alpha for the number of comparisons to be performed
                ngroups = len(groups)
                alpha = 0.05 / ngroups * (ngroups - 1) / 2

                # Get unique pairwise comparisons
                paircomp = list(combinations(np.unique(grouping), 2))

                # Set number of subplots for comparison
                if len(paircomp) == 3:
                    varfigs[f"{key}_posthocs"], axes = plt.subplots(
                        2, 3, figsize=(11, 6)
                    )

                elif len(paircomp) == 6:
                    varfigs[f"{key}_posthocs"], axes = plt.subplots(
                        4, 3, figsize=(11, 12)
                    )

                else:
                    print("I am not ready for so many plots. Figure it out.")
                axes = axes.flat
                for pairi, pair in enumerate(paircomp):
                    # Get pair key word
                    pairkw = f"{str(pair[0])}_{str(pair[1])}"

                    # Run post-hoc analysis
                    cont_comp[key]["post_hoc_np_ttest2"] = {}
                    nonparam_ttest2 = spm1d.stats.nonparam.ttest2(
                        groups[pair[0]], groups[pair[1]]
                    )
                    cont_comp[key]["post_hoc_np_ttest2"][pairkw] = (
                        nonparam_ttest2.inference(
                            alpha=alpha, two_tailed=True, iterations=500
                        )
                    )

                    # Vis
                    if pairi <= 2:
                        axi = pairi
                    else:
                        axi = pairi + 6

                    # NOTE THIS ASSUMES THAT THE ORDER OF THE COLOURS MATCHES THE ORDER OF THE LABELS
                    spm1d.plot.plot_mean_sd(
                        groups[pair[0]],
                        ax=axes[axi],
                        linecolor=colours[pair[0]],
                        facecolor=colours[pair[0]],
                    )
                    spm1d.plot.plot_mean_sd(
                        groups[pair[1]],
                        ax=axes[axi],
                        linecolor=colours[pair[0]],
                        facecolor=colours[pair[1]],
                    )
                    axes[pairi].set_title(str(pair))

                    pairkw = f"{str(pair[0])}_{str(pair[1])}"
                    plot_spm_test(
                        cont_comp[key]["post_hoc_np_ttest2"][pairkw],
                        title=str(pair),
                        ax=axes[axi + 3],
                    )

                varfigs[f"{key}_posthocs"].suptitle(f"{title_kword}_{key}")
                plt.tight_layout()

    return cont_comp, varfigs


def write_0Dposthoc_statstr(
    posthoctable, contrastvalue, withinfactor, withinfactorvalue
):
    """
    Generate a string representation of post-hoc test statistics for a given contrast and within-factor value.

    Parameters:
    posthoctable (pd.DataFrame): The DataFrame containing the post-hoc test results.
    contrastvalue (str): The contrast value to filter the post-hoc table.
    withinfactor (str): The within-subjects factor to filter the post-hoc table.
    withinfactorvalue (str): The value of the within-subjects factor to filter the post-hoc table.

    Returns:
    str: A string representation of the post-hoc test statistics, including t-value, p-value, and Cohen's d.
    """

    t = np.round(
        posthoctable["T"]
        .loc[
            (posthoctable["Contrast"] == contrastvalue)
            & (posthoctable[withinfactor] == withinfactorvalue)
        ]
        .values[0],
        2,
    )
    d = np.round(
        posthoctable["cohen"]
        .loc[
            (posthoctable["Contrast"] == contrastvalue)
            & (posthoctable[withinfactor] == withinfactorvalue)
        ]
        .values[0],
        2,
    )

    ci = [
        posthoctable["esci95_low"]
        .loc[
            (posthoctable["Contrast"] == contrastvalue)
            & (posthoctable[withinfactor] == withinfactorvalue)
        ]
        .values[0],
        posthoctable["esci95_up"]
        .loc[
            (posthoctable["Contrast"] == contrastvalue)
            & (posthoctable[withinfactor] == withinfactorvalue)
        ]
        .values[0],
    ]

    if (
        posthoctable["p-corr"]
        .loc[
            (posthoctable["Contrast"] == contrastvalue)
            & (posthoctable[withinfactor] == withinfactorvalue)
        ]
        .values[0]
        < 0.001
    ):
        p = "< 0.001"
    else:
        p = np.round(
            posthoctable["p-corr"]
            .loc[
                (posthoctable["Contrast"] == contrastvalue)
                & (posthoctable[withinfactor] == withinfactorvalue)
            ]
            .values[0],
            3,
        )

    return f"t = {t}, p = {p}, d = {d}[{np.round(ci[0], 2)}, {np.round(ci[1], 2)}]"


def write_0DmixedANOVA_statstr(
    mixed_anovatable,
    between="",
    within="",
    betweenlabel="",
    withinlabel="",
    write_between=True,
    write_within=True,
    write_interaction=True,
):
    """
    Write a formatted string summarizing the results of a mixed ANOVA with one between-subjects factor and
     one within-subjects factor.

    Parameters:
    mixed_anovatable (pd.DataFrame): DataFrame containing the ANOVA results. Output of penguoin mixed_anova.
    between (str): Name of the between-subjects factor.
    within (str): Name of the within-subjects factor.
    betweenlabel (str, optional): Label for the between-subjects factor. Defaults to the value of 'between'.
    withinlabel (str, optional): Label for the within-subjects factor. Defaults to the value of 'within'.

    Returns:
    statstr (str): A formatted string summarizing the ANOVA results.
    """

    # Get factor labels or set them to factor names if not provided
    if betweenlabel == "":
        betweenlabel = between
    if withinlabel == "":
        withinlabel = within

    statstr = ""

    if write_between:
        if (
            mixed_anovatable["p-unc"].loc[mixed_anovatable["Source"] == between].values
            < 0.001
        ):
            statstr += f"{betweenlabel}: F = {np.round(mixed_anovatable['F'].values[0], 2)}, p < 0.001"
        else:
            statstr += (
                f"{betweenlabel}: F = {np.round(mixed_anovatable['F'].values[0], 2)}, "
                f"p = {np.round(mixed_anovatable['p-unc'].values[0], 3)}"
            )

    if write_within:
        if (
            mixed_anovatable["p-unc"].loc[mixed_anovatable["Source"] == within].values
            < 0.001
        ):
            statstr += f"; {withinlabel}: F = {np.round(mixed_anovatable['F'].values[1], 2)}, p < 0.001"
        else:
            statstr += (
                f"; {withinlabel}: F = {np.round(mixed_anovatable['F'].values[1], 2)}, "
                f"p = {np.round(mixed_anovatable['p-unc'].values[1], 3)}"
            )

    if write_interaction:
        if (
            mixed_anovatable["p-unc"]
            .loc[mixed_anovatable["Source"] == "Interaction"]
            .values
            < 0.001
        ):
            statstr += (
                f"; {betweenlabel}x{withinlabel}: F = {np.round(mixed_anovatable['F'].values[2], 2)}, "
                f"p < 0.001"
            )
        else:
            statstr += (
                f"; {betweenlabel}x{withinlabel}: F = {np.round(mixed_anovatable['F'].values[2], 2)}, "
                f"p = {np.round(mixed_anovatable['p-unc'].values[2], 2)}"
            )

    return statstr


def write_spm_stats_str(spmobj, mode="full"):
    """
    Generate a string representation of SPM (Statistical Parametric Mapping) statistics.

    Parameters:
    spmobj (object): The SPM object containing the statistical results.
    mode (str, optional): The mode of the output string. Must be one of 'full', 'stat', or 'p'. Defaults to 'full'.

    Returns:
    str: A string representation of the SPM statistics.

    Raises:
    ValueError: If the mode is not one of 'full', 'stat', or 'p'.
    """

    # Make sure mode is full, stat or p
    if mode not in ["full", "stat", "p"]:
        raise ValueError("mode must be either full, stat or p")

    # Initialise statsstr
    statsstr = ""

    # Add stat value
    if mode == "full" or mode == "stat":
        statsstr = f"{np.round(spmobj.zstar, 2)}"

    # Add p value
    if mode == "full" or mode == "p":
        if len(spmobj.p) == 1:
            if spmobj.p[0] < 0.001:
                statsstr += ", p < 0.001"
            else:
                statsstr += f", p = {np.round(spmobj.p[0], 3)}"
        elif len(spmobj.p) > 1:
            statsstr += ", p = ["
            for i, p in enumerate(spmobj.p):
                if i > 0:
                    statsstr += ", "
                if p < 0.001:
                    statsstr += "< 0.001"
                else:
                    statsstr += f"{np.round(p, 3)}"
            statsstr += "]"

    return statsstr


def plot_spm_test(spm_obj, title, ax=None):
    """ """
    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    spm_obj.plot(ax=ax)
    spm_obj.plot_threshold_label(ax=ax, fontsize=8)
    spm_obj.plot_p_values(ax=ax, size=10)
    ax.set_title(title)

    if return_fig:
        return fig
