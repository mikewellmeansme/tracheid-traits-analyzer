import app.plotting

from pandas import DataFrame, unique, merge, concat
from typing import Dict, List, Optional, Tuple, Union
from zhutils.approximators import Approximator
from zhutils.tracheids import Tracheids


class TracheidTraits:
    __data__: DataFrame
    __trees__: List[str]
    __names__: List[str]

    def __init__(self, tracheids: Tracheids) -> None:
        self.__names__ = ['TRW', '№', 'Dmax', 'Dmean', 'CWTmax', 'CWTmean']

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

    # ----------------------------------------------------------------------
    # Add plotting methods to TracheidTraits
    plot = app.plotting.plot_tracheid_traits
    hist = app.plotting.hist_tracheid_traits
    qqplot = app.plotting.qqplot_tracheid_traits
    scatter = app.plotting.scatter_tracheid_traits
    plot_sample_depth = app.plotting.plot_tracheid_traits_sample_depth

    # ----------------------------------------------------------------------

    def apply_model(
            self,
            x_trait: str,
            y_trait: str,
            approximator: Approximator,
            approximator_kws: Optional[Dict] = None
    ) -> Tuple[DataFrame, Dict[str, List[float]]]:

        self.__check_trait__(x_trait)
        self.__check_trait__(y_trait)

        approximator_kws = approximator_kws or {}

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
    
    def get_sample_depth(self) -> DataFrame:
        result = self.__data__.groupby('Year').count().reset_index()
        result = result[['Year', 'Tree']].rename(columns={'Tree': 'Depth'})
        return result
    
    def __check_trait__(self, trait: str) -> None:
        if trait not in self.__names__:
            raise KeyError(f'Trait name "{trait}" is not in the list of trait names!')

    def __check_tree__(self, tree: str) -> None:
        if tree not in self.__trees__:
            raise KeyError(f'Tree name "{tree}" is not in the list of tree names!')
