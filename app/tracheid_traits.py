import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.figure import Figure, Axes
from pandas import DataFrame, unique, merge, concat
from scipy import stats
from sklearn.metrics import r2_score
from typing import Dict, List, Optional, Tuple, Union
from zhutils.approximators import Approximator
from zhutils.tracheids import Tracheids


class TracheidTraits:
    __data__: DataFrame
    __trees__: List[str]
    __names__: List[str] = ['TRW', '№', 'Dmax', 'Dmean', 'CWTmax', 'CWTmean']

    def __init__(self, tracheids: Tracheids) -> None:
        max_values = tracheids.data.groupby(['Tree', 'Year']).max().reset_index()
        mean_values = tracheids.data.groupby(['Tree', 'Year']).mean().reset_index()
        data = max_values[['Tree', 'Year', 'TRW', '№']].copy()
        data['Dmax'] = max_values['Dmean']
        data['CWTmax'] = max_values['CWTmean']
        data['Dmean'] = mean_values['Dmean']
        data['CWTmean'] = mean_values['CWTmean']

        self.__data__ = data
        self.__trees__ = unique(data['Tree']).tolist()

    def describe(self, trait: str) -> DataFrame:
        self.__check_trait__(trait)
        d = self.__data__.groupby('Tree').describe().unstack().reset_index()

        description = d[d['level_0'] == trait].pivot(index='Tree', values=0, columns='level_1')
        skewness = self.__data__.groupby('Tree').skew().reset_index()

        # FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated;
        # in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
        kurtosis = self.__data__.groupby('Tree').apply(DataFrame.kurt).reset_index()

        result = \
            description.\
            reset_index().\
            merge(skewness[['Tree', trait]].rename(columns={trait: 'Skewness'}), how='inner', on='Tree').\
            merge(kurtosis[['Tree', trait]].rename(columns={trait: 'Kurtosis'}), how='inner', on='Tree')
        return result

    def rename_trees(self, trees: List[str]) -> None:
        if len(trees) != len(self.__trees__):
            raise Exception('The number of tree names is not equal to the number of trees!')
        self.__data__ = self.__data__.replace(self.__trees__, trees)
        self.__trees__ = trees

    def add_trait(self, name: str, data: DataFrame) -> None:
        if name in self.__names__:
            raise Exception(f"Trait with name '{name}' already exists!")
        if not {'Tree', 'Year', name} <= set(data.columns):
            raise Exception(f"Given DataFrame does not have all of the following columns: 'Tree', 'Year', '{name}' ")
        self.__data__ = merge(self.__data__, data[['Tree', 'Year', name]], on=('Tree', 'Year'), how='left')
        self.__names__.append(name)

    def get_traits(
            self,
            traits: Optional[Union[str, List[str]]] = None,
            trees: Optional[Union[str, List[str]]] = None
    ) -> DataFrame:

        result = self.__data__.copy()

        if isinstance(traits, str):
            self.__check_trait__(traits)
            result = result[['Year', 'Tree', traits]]
        elif isinstance(traits, List):
            for trait in traits:
                self.__check_trait__(trait)
            result = result[['Year', 'Tree', *traits]]

        if isinstance(trees, str):
            self.__check_tree__(trees)
            result = result[result['Tree'] == trees]
        elif isinstance(trees, List):
            for tree in trees:
                self.__check_tree__(tree)
            result = result[result['Tree'].isin(trees)]

        return result.reset_index(drop=True)
    
    def get_trees(self) -> List[str]:
        return self.__trees__.copy()

    def hist(
            self,
            trait: str,
            axes: Optional[List[Axes]] = None,
            subplots_kws: Optional[Dict] = None
    ) -> Tuple[Figure, List[Axes]]:

        self.__check_trait__(trait)
        n = len(self.__trees__)
        fig, axes = self.__get_subplots__(1, n, axes, subplots_kws)
        for i, group_data in enumerate(self.__data__.groupby('Tree')):
            tree, df = group_data
            sns.set_style("white")
            sns.histplot(df[trait], ax=axes[i], color='black', kde=True, stat='probability')
            axes[i].set_title(tree)

        return fig, axes

    def qqplot(
            self,
            trait: str,
            dist: str = 'norm',
            axes: Optional[List[Axes]] = None,
            subplots_kws: Optional[Dict] = None
    ) -> Tuple[Figure, List[Axes]]:

        self.__check_trait__(trait)
        n = len(self.__trees__)
        fig, axes = self.__get_subplots__(1, n, axes, subplots_kws)
        for i, group_data in enumerate(self.__data__.groupby('Tree')):
            tree, df = group_data
            stats.probplot(df[trait], dist=dist, plot=axes[i])
            axes[i].set_title(tree)

        return fig, axes

    def scatter(
            self,
            x_trait: str,
            y_trait: str,
            trees: Optional[List[str]] = None,
            labels: Optional[Dict[str, str]] = None,
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

        self.__check_trait__(x_trait)
        self.__check_trait__(y_trait)

        trees = trees if trees else sorted(self.__trees__)
        for tree in trees:
            self.__check_tree__(tree)
        
        if labels:
            if not set(trees) <= set(labels.keys()):
                raise KeyError(f'The given labels do not correspond to the all given trees!')

        approximator_kws = {} if approximator_kws is None else approximator_kws
        plot_kws = {} if plot_kws is None else plot_kws
        scatter_kws = {} if scatter_kws is None else scatter_kws

        n = len(trees)
        fig, axes = self.__get_subplots__(1, n, axes, subplots_kws)

        groups = self.__data__.groupby('Tree')

        for i, tree in enumerate(trees):
            df = groups.get_group(tree)
            if approximator is not None:
                approximator.fit(df[x_trait], df[y_trait], **approximator_kws)
                equation = approximator.get_equation(2)
                
                r2 = f'\n$R^{{2}}={r2_score(df[y_trait], approximator.predict(df[x_trait])):.2f}$' if show_r2 else ''

                x = np.arange(min(df[x_trait]), max(df[x_trait]), 1/len(df[x_trait]))
                y = approximator.predict(x)
                label = labels[tree] if labels else equation
                axes[i].plot(x, y, label= label + r2, **plot_kws)
                axes[i].legend(frameon=False)
            axes[i].scatter(df[x_trait], df[y_trait], **scatter_kws)
            axes[i].set_title(tree)
            axes[i].set_xlabel(xlabel if xlabel else x_trait)
            axes[i].set_ylabel(ylabel if ylabel else y_trait)

        return fig, axes

    def apply_model(
            self,
            x_trait: str,
            y_trait: str,
            approximator: Approximator,
            approximator_kws: Optional[Dict] = None
    ) -> Tuple[DataFrame, Dict[str, List[float]]]:

        self.__check_trait__(x_trait)
        self.__check_trait__(y_trait)

        approximator_kws = {} if approximator_kws is None else approximator_kws

        result = DataFrame({
            'Tree': [],
            'Year': [],
            f'{y_trait}_model': []
        })
        coeffs = dict()
        for i, group_data in enumerate(self.__data__.groupby('Tree')):
            tree, df = group_data
            approximator.fit(df[x_trait], df[y_trait], **approximator_kws)
            modeled_values = DataFrame({
                'Tree': df['Tree'],
                'Year': df['Year'],
                f'{y_trait}_model': approximator.predict(df[x_trait])
            })
            result = concat([result, modeled_values])
            coeffs[tree] = approximator.coeffs

        return result.reset_index(drop=True), coeffs

    @staticmethod
    def __get_subplots__(
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
            fig = axes[0].figure
        return fig, axes

    def __check_trait__(self, trait: str) -> None:
        if trait not in self.__names__:
            raise KeyError(f'Trait name "{trait}" is not in the list of trait names!')

    def __check_tree__(self, tree: str) -> None:
        if tree not in self.__trees__:
            raise KeyError(f'Tree name "{tree}" is not in the list of tree names!')
