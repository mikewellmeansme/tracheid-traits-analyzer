import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from pandas import DataFrame, unique, merge
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from zhutils.approximators import Approximator
from zhutils.tracheids import Tracheids


@dataclass
class TracheidTraitsDescription:
    trw: DataFrame
    cell_amount: DataFrame
    d_max: DataFrame
    d_mean: DataFrame
    cwt_max: DataFrame
    cwt_mean: DataFrame


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

    def describe(self) -> TracheidTraitsDescription:
        d = self.__data__.groupby('Tree').describe().unstack().reset_index()
        descriptions = dict()
        for trait in self.__names__:
            description = d[d['level_0'] == trait].pivot(index='Tree', values=0, columns='level_1')
            skewness = self.__data__.groupby('Tree').skew().reset_index()
            kurtosis = self.__data__.groupby('Tree').apply(DataFrame.kurt).reset_index()
            descriptions[trait] = \
                description.\
                reset_index().\
                merge(skewness[['Tree', trait]].rename(columns={trait: 'Skewness'}), how='inner', on='Tree').\
                merge(kurtosis[['Tree', trait]].rename(columns={trait: 'Kurtosis'}), how='inner', on='Tree')
        result = TracheidTraitsDescription(*descriptions.values())
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

    def hist(self, trait: str) -> Tuple[Figure, Axes]:
        self.__check_trait__(trait)
        n = len(self.__trees__)
        fig, ax = self.__get_subplots__(1, n)
        for i, group_data in enumerate(self.__data__.groupby('Tree')):
            tree, df = group_data
            sns.set_style("white")
            sns.histplot(df[trait], ax=ax[i], color='black', kde=True, stat='probability')
            ax[i].set_title(tree)

        return fig, ax

    def qqplot(self, trait: str, dist: str = 'norm') -> Tuple[Figure, Axes]:
        self.__check_trait__(trait)
        n = len(self.__trees__)
        fig, ax = self.__get_subplots__(1, n)
        for i, group_data in enumerate(self.__data__.groupby('Tree')):
            tree, df = group_data
            stats.probplot(df[trait], dist=dist, plot=ax[i])
            ax[i].set_title(tree)

        return fig, ax

    def scatter(
            self,
            x_trait: str,
            y_trait: str,
            approximator: Optional[Approximator] = None,
            approximator_kws: Optional[Dict] = None,
            plot_kws: Optional[Dict] = None,
            scatter_kws: Optional[Dict] = None,
            axes: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        # TODO: custom axes shape

        self.__check_trait__(x_trait)
        self.__check_trait__(y_trait)

        approximator_kws = {} if approximator_kws is None else approximator_kws
        plot_kws = {} if plot_kws is None else plot_kws
        scatter_kws = {} if scatter_kws is None else scatter_kws

        n = len(self.__trees__)
        if axes is None:
            fig, axes = self.__get_subplots__(1, n)
        else:
            fig = axes.figure

        axes[0].set_ylabel(y_trait)

        for i, group_data in enumerate(self.__data__.groupby('Tree')):
            tree, df = group_data
            if approximator is not None:
                approximator.fit(df[x_trait], df[y_trait], **approximator_kws)
                equation = approximator.get_equation()
                x = sorted(df[x_trait])
                y = approximator.predict(x)
                axes[i].plot(x, y, label=equation, **plot_kws)
                axes[i].legend(frameon=False)
            axes[i].scatter(df[x_trait], df[y_trait], **scatter_kws)
            axes[i].set_title(tree)
            axes[i].set_xlabel(x_trait)

        return fig, axes

    @staticmethod
    def __get_subplots__(
            nrows: int,
            ncols: int,
            sharex: str = 'all',
            sharey: str = 'all',
            **subplots_kwargs
    ) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            sharex=sharex,
            sharey=sharey,
            figsize=(ncols * 3, nrows * 3),
            **subplots_kwargs
        )
        return fig, ax

    def __check_trait__(self, trait: str) -> None:
        if trait not in self.__names__:
            raise KeyError(f'Trait name "{trait}" is not in the list of trait names!')

    def __check_tree__(self, tree: str) -> None:
        if tree not in self.__trees__:
            raise KeyError(f'Tree name "{tree}" is not in the list of tree names!')
