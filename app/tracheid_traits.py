import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from pandas import DataFrame, unique
from scipy import stats
from typing import List, Tuple
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
    data: DataFrame
    trees: List[str]
    names: List[str] = ['TRW', '№', 'Dmax', 'Dmean', 'CWTmax', 'CWTmean']

    def __init__(self, tracheids: Tracheids) -> None:
        max_values = tracheids.data.groupby(['Tree', 'Year']).max().reset_index()
        mean_values = tracheids.data.groupby(['Tree', 'Year']).mean().reset_index()
        data = max_values[['Tree', 'Year', 'TRW', '№']]
        data['Dmax'] = max_values['Dmean']
        data['CWTmax'] = max_values['CWTmean']
        data['Dmean'] = mean_values['Dmean']
        data['CWTmean'] = mean_values['CWTmean']

        self.data = data
        self.trees = unique(data['Tree']).tolist()

    def describe(self) -> TracheidTraitsDescription:
        d = self.data.groupby('Tree').describe().unstack().reset_index()
        descriptions = dict()
        for trait in self.names:
            description = d[d['level_0'] == trait].pivot(index='Tree', values=0, columns='level_1')
            skewness = self.data.groupby('Tree').skew().reset_index()
            kurtosis = self.data.groupby('Tree').apply(DataFrame.kurt).reset_index()
            descriptions[trait] = \
                description.\
                reset_index().\
                merge(skewness[['Tree', trait]].rename(columns={trait: 'Skewness'}), how='inner', on='Tree').\
                merge(kurtosis[['Tree', trait]].rename(columns={trait: 'Kurtosis'}), how='inner', on='Tree')
        result = TracheidTraitsDescription(*descriptions.values())
        return result

    def rename_trees(self, trees: List[str]) -> None:
        if len(trees) == len(self.trees):
            self.data = self.data.replace(self.trees, trees)
            self.trees = trees
        else:
            raise Exception('The number of tree names is not equal to the number of trees!')

    def hist(self, trait: str) -> Tuple[Figure, Axes]:
        self.__check_trait__(trait)
        n = len(self.trees)
        fig, ax = plt.subplots(
            ncols=n,
            sharey='all',
            sharex='all',
            figsize=(n*3, 3)
        )
        for i, group_data in enumerate(self.data.groupby('Tree')):
            tree, df = group_data
            sns.set_style("white")
            sns.histplot(df[trait], ax=ax[i], color='black', kde=True, stat='probability')
            ax[i].set_title(tree)

        return fig, ax

    def qqplot(self, trait: str, dist: str = 'norm') -> Tuple[Figure, Axes]:
        self.__check_trait__(trait)
        n = len(self.trees)
        fig, ax = plt.subplots(
            ncols=n,
            sharey='all',
            sharex='all',
            figsize=(n * 3, 3)
        )
        for i, group_data in enumerate(self.data.groupby('Tree')):
            tree, df = group_data
            stats.probplot(df[trait], dist=dist, plot=ax[i])
            ax[i].set_title(tree)
            i += 1

        return fig, ax

    def __check_trait__(self, trait: str) -> None:
        if trait not in self.names:
            raise KeyError(f'Trait name "{trait}" is not in the list of trait names!')
