from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.figure import Figure, Axes
from scipy import stats
from sklearn.metrics import r2_score
from typing import Dict, List, Optional, Tuple, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.tracheid_traits import TracheidTraits

from zhutils.approximators import Approximator


def plot_tracheid_traits(
        data: TracheidTraits,
        trait: str,
        *,
        trees: Optional[List[str]] = None,
        show_individual: bool = True,
        show_mean: bool = True,
        show_std: bool = True,
        plot_kws: Optional[Dict] = None,
        mean_kws: Optional[Dict] = None,
        std_kws: Optional[Dict] = None,
        axes: Optional[Axes] = None,
        subplots_kws: Optional[Dict] = None
) -> Tuple[Figure, Axes]:
    
    data.__check_trait__(trait)
    
    trees = trees or data.__trees__

    for tree in trees:
        data.__check_tree__(tree)
    
    plot_kws = plot_kws or {}
    mean_kws = mean_kws or {}
    std_kws = std_kws or {}

    data = data.__data__[data.__data__['Tree'].isin(trees)]
    groups_per_tree = data.groupby('Tree')

    fig, axes = get_subplots(1, 1, axes, subplots_kws)

    if show_individual:
        for tree in trees:
            df = groups_per_tree.get_group(tree).reset_index()
            axes.plot(df['Year'], df[trait], label=tree, **plot_kws)
    
    groups_per_year = data.groupby('Year')
    if show_mean:
        mean_df = groups_per_year.mean().reset_index()
        axes.plot(mean_df['Year'], mean_df[trait], **mean_kws)
        if show_std:
            std_df = groups_per_year.std().reset_index()
            axes.fill_between(std_df['Year'], mean_df[trait]-std_df[trait], mean_df[trait]+std_df[trait], **std_kws)
    
    axes.legend()
    axes.set_title(trait)
    axes.set_xlabel('Year')

    return fig, axes


def hist_tracheid_traits(
        data : TracheidTraits,
        trait: str,
        *,
        trees: Optional[List[str]] = None,
        axes: Optional[List[Axes]] = None,
        subplots_kws: Optional[Dict] = None,
        histplot_kws:  Optional[Dict] = None
) -> Tuple[Figure, List[Axes]]:

    data.__check_trait__(trait)

    trees = trees or sorted(data.__trees__)
    histplot_kws = histplot_kws or {'color': 'black', 'kde': True, 'stat':'probability'}

    n = len(trees)
    fig, axes = get_subplots(1, n, axes, subplots_kws)
    axes = axes if n > 1 else [axes]

    groups = data.__data__.groupby('Tree')

    for i, tree in enumerate(trees):
        df = groups.get_group(tree)
        sns.histplot(df[trait], ax=axes[i], **histplot_kws)
        axes[i].set_title(tree)

    return fig, axes if n > 1 else axes[0]


def qqplot_tracheid_traits(
        data: TracheidTraits,
        trait: str,
        *,
        trees: Optional[List[str]] = None,
        axes: Optional[List[Axes]] = None,
        subplots_kws: Optional[Dict] = None,
        probplot_kws: Optional[Dict] = None
) -> Tuple[Figure, List[Axes]]:

    data.__check_trait__(trait)

    trees = trees or sorted(data.__trees__)
    probplot_kws = probplot_kws or {'dist': 'norm'}

    n = len(trees)
    fig, axes = get_subplots(1, n, axes, subplots_kws)
    axes = axes if n > 1 else [axes]

    groups = data.__data__.groupby('Tree')

    for i, tree in enumerate(trees):
        df = groups.get_group(tree)
        stats.probplot(df[trait], plot=axes[i], **probplot_kws)
        axes[i].set_title(tree)

    return fig, axes if n > 1 else axes[0]


def scatter_tracheid_traits(
        data: TracheidTraits,
        x_trait: str,
        y_trait: str,
        *,
        trees: Optional[List[str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        approximator: Optional[Approximator] = None,
        approximator_kws: Optional[Dict] = None,
        plot_kws: Optional[Dict] = None,
        scatter_kws: Optional[Dict] = None,
        axes: Optional[List[Axes]] = None,
        subplots_kws: Optional[Dict] = None,
        show_r2: bool = True
) -> Tuple[Figure, List[Axes]]:

    data.__check_trait__(x_trait)
    data.__check_trait__(y_trait)

    trees = trees or sorted(data.__trees__)

    for tree in trees:
        data.__check_tree__(tree)

    approximator_kws = approximator_kws or {}
    plot_kws = plot_kws or {}
    scatter_kws = scatter_kws or {}

    n = len(trees)
    fig, axes = get_subplots(1, n, axes, subplots_kws)
    axes = axes if n > 1 else [axes]

    groups = data.__data__.groupby('Tree')

    plot_label = plot_kws.pop('label', None)

    for i, tree in enumerate(trees):
        df = groups.get_group(tree)
        if approximator is not None:
            approximator.fit(df[x_trait], df[y_trait], **approximator_kws)
            equation = approximator.get_equation(2)
            r2 = f'\n$R^{{2}}={r2_score(df[y_trait], approximator.predict(df[x_trait])):.2f}$' if show_r2 else ''
            x = np.arange(min(df[x_trait]), max(df[x_trait]), 1/len(df[x_trait]))
            y = approximator.predict(x)
            
            label = plot_label if plot_label else equation

            axes[i].plot(x, y, label=label + r2, **plot_kws)
        
        axes[i].scatter(df[x_trait], df[y_trait], **scatter_kws)
        axes[i].set_title(tree)
        axes[i].set_xlabel(xlabel if xlabel else x_trait)
        axes[i].set_ylabel(ylabel if ylabel else y_trait)
        axes[i].legend(frameon=False)

    return fig, axes if n > 1 else axes[0]


def plot_tracheid_traits_sample_depth(
        data: TracheidTraits,
        axes: Optional[Axes] = None,
        subplots_kws: Optional[Dict] = None,
        barplot_kws: Optional[Dict] = None
    ) -> Tuple[Figure, Axes]:

    barplot_kws = barplot_kws or {}

    sample_depth = data.get_sample_depth()
    fig, axes = get_subplots(1, 1, axes, subplots_kws)
    axes.bar(sample_depth['Year'], sample_depth['Depth'], **barplot_kws)
    axes.set_xlabel('Year')
    axes.set_ylabel('Sample depth (trees)')

    return fig, axes


def get_subplots(
        nrows: int,
        ncols: int,
        axes: Optional[List[Axes]] = None,
        subplots_kws: Optional[Dict] = None
) -> Tuple[Figure, List[Axes]]:
    if axes is None:

        subplots_kws = {
            'sharex': 'all',
            'sharey': 'all',
            'figsize': (ncols * 3, nrows * 3)
        } if subplots_kws is None else subplots_kws

        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            **subplots_kws
        )
    else:
        if isinstance(axes, Iterable):
            if len(axes.shape) == 2:
                fig = axes[0, 0].figure
            else:
                fig = axes[0].figure
        else:
            fig = axes.figure
    return fig, axes