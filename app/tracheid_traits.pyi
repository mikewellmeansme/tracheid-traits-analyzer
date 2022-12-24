from matplotlib.figure import Figure, Axes
from pandas import DataFrame
from typing import Dict, List, Optional, Tuple, Union
from zhutils.approximators import Approximator
from zhutils.tracheids import Tracheids


class TracheidTraits:
    __data__: DataFrame
    __trees__: List[str]
    __names__: List[str]

    def __init__(self, tracheids: Tracheids) -> None: ... 

    def describe(self, trait: str) -> DataFrame: ...

    def rename_trees(self, trees: List[str]) -> None: ...

    def add_trait(self, name: str, data: DataFrame) -> None: ...

    def get_traits(
            self,
            traits: Optional[Union[str, List[str]]],
            trees: Optional[Union[str, List[str]]]
    ) -> DataFrame: ...
    
    def get_trees(self) -> List[str]: ...
    
    def plot(
            self,
            trait: str,
            *,
            trees: Optional[List[str]],
            show_individual: bool,
            show_mean: bool,
            show_std: bool,
            plot_kws: Optional[Dict],
            mean_kws: Optional[Dict],
            std_kws: Optional[Dict],
            axes: Optional[Axes],
            subplots_kws: Optional[Dict]
    ) -> Tuple[Figure, Axes]: ... 

    def hist(
            self,
            trait: str,
            axes: Optional[List[Axes]],
            subplots_kws: Optional[Dict]
    ) -> Tuple[Figure, List[Axes]]: ...

    def qqplot(
            self,
            trait: str,
            dist: str,
            axes: Optional[List[Axes]],
            subplots_kws: Optional[Dict]
    ) -> Tuple[Figure, List[Axes]]: ...

    def scatter(
            self,
            x_trait: str,
            y_trait: str,
            *,
            trees: Optional[List[str]],
            xlabel: Optional[str],
            ylabel: Optional[str],
            approximator: Optional[Approximator],
            approximator_kws: Optional[Dict],
            plot_kws: Optional[Dict],
            scatter_kws: Optional[Dict],
            axes: Optional[List[Axes]],
            subplots_kws: Optional[Dict],
            show_r2: bool
    ) -> Tuple[Figure, List[Axes]]: ...

    def apply_model(
            self,
            x_trait: str,
            y_trait: str,
            approximator: Approximator,
            approximator_kws: Optional[Dict]
    ) -> Tuple[DataFrame, Dict[str, List[float]]]: ...
    
    def get_sample_depth(self) -> DataFrame: ...
    
    def plot_sample_depth(
            self,
            axes: Optional[Axes],
            subplots_kws: Optional[Dict],
            barplot_kws: Optional[Dict]
        ) -> Tuple[Figure, Axes]: ...

    def __check_trait__(self, trait: str) -> None: ...

    def __check_tree__(self, tree: str) -> None: ... 
