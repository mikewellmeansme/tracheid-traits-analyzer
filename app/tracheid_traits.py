from dataclasses import dataclass
from pandas import DataFrame
from typing import List
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
    trait_names: List = ['TRW', '№', 'Dmax', 'Dmean', 'CWTmax', 'CWTmean']

    def __init__(self, tracheids: Tracheids) -> None:
        max_values = tracheids.data.groupby(['Tree', 'Year']).max().reset_index()
        mean_values = tracheids.data.groupby(['Tree', 'Year']).mean().reset_index()
        data = max_values[['Tree', 'Year', 'TRW', '№']]
        data['Dmax'] = max_values['Dmean']
        data['CWTmax'] = max_values['CWTmean']
        data['Dmean'] = mean_values['Dmean']
        data['CWTmean'] = mean_values['CWTmean']

        self.data = data

    def describe(self) -> TracheidTraitsDescription:
        d = self.data.groupby('Tree').describe().unstack().reset_index()
        descriptions = dict()
        for trait in self.trait_names:
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