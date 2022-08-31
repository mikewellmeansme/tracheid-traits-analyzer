import matplotlib.pyplot as plt
import pandas as pd

from tracheid_traits import TracheidTraits
from zhutils.approximators import Polynomial
from zhutils.correlation import dropna_pearsonr
from zhutils.dataframes import DailyDataFrame
from zhutils.tracheids import Tracheids

from exponential_approximator import Exponential

# TODO: deal with fonts in hists and qqplots
plt.rcParams['font.size'] = '16'
plt.rcParams['font.family'] = 'Times New Roman'


def main(tracheid_path, climate_path):
    climate_df = DailyDataFrame(pd.read_csv(climate_path))
    tr = Tracheids(tracheid_path, tracheid_path, [])
    tr_t = TracheidTraits(tr)
    result_plots = dict()
    result_tables = dict()

    fig, ax = tr_t.scatter(
        '№',
        'TRW',
        approximator=Polynomial(),
        approximator_kws={'deg': 1},
        plot_kws={'color': 'black', 'linewidth': 2},
        xlabel='N',
        scatter_kws={'color': 'white', 'edgecolor': 'gray'},
        subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)}
    )
    result_plots['TRW_N'] = fig

    d_min = tr.data.min()['Dmean']*0.95

    fig, axes = tr_t.scatter(
        '№',
        'Dmax',
        approximator=Exponential(d_min),
        plot_kws={'color': 'black', 'linewidth': 2, 'label': 'Dmax(N)'},
        scatter_kws={'color': 'white', 'edgecolor':'red', 'label': 'Dmax'}, 
        subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)}
    )
    fig, axes = tr_t.scatter(
        '№',
        'Dmean',
        approximator=Exponential(d_min),
        plot_kws={'color': 'black', 'linewidth': 2, 'linestyle': '--', 'label': 'Dmean(N)'},
        axes=axes,
        scatter_kws={'color': 'white', 'edgecolor':'blue', 'label': 'Dmean'},
        xlabel='N',
        ylabel='D (μm)'
    )
    for ax in axes:
        ax.legend(frameon=True)
    
    result_plots['D_N'] = fig

    cwt_min = tr.data.min()['CWTmean']*0.95

    fig, axes = tr_t.scatter(
        '№',
        'CWTmax',
        approximator=Exponential(cwt_min),
        plot_kws={'color': 'black', 'linewidth': 2, 'label': 'CWTmax(N)'},
        scatter_kws={'color': 'white', 'edgecolor':'red', 'label': 'CWTmax'}, 
        subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)}
    )
    fig, axes = tr_t.scatter(
        '№',
        'CWTmean',
        approximator=Exponential(cwt_min),
        plot_kws={'color': 'black', 'linewidth': 2, 'linestyle': '--', 'label': 'CWTmean(N)'},
        axes=axes,
        scatter_kws={'color': 'white', 'edgecolor':'blue', 'label': 'CWTmean'},
        xlabel='N',
        ylabel='CWT (μm)'
    )
    for ax in axes:
        ax.legend(frameon=True)
    
    result_plots['CWT_N'] = fig

    for trait in ['Dmean', 'Dmax', 'CWTmax', 'CWTmean']:
        df, coeffs = tr_t.apply_model('№', trait, Exponential(d_min if 'D' in trait else cwt_min))
        tr_t.add_trait(f'{trait}_model', df)
        loc_df = tr_t.get_traits([trait, f'{trait}_model'])
        loc_df[f'{trait}_ind'] = loc_df[trait] / loc_df[f'{trait}_model']
        tr_t.add_trait(f'{trait}_ind', loc_df)
        result_tables[f'{trait}_coeffs'] = pd.DataFrame(coeffs)
    
    result_tables['TracheidTraits'] = tr_t.get_traits()
    
    for trait in ['Dmean', 'Dmax', 'CWTmax', 'CWTmean', 'Dmean_ind', 'Dmax_ind', 'CWTmax_ind', 'CWTmean_ind']:
        def compare(df, index):
            r, p = dropna_pearsonr(df[index], df[trait])
            return r, p

        data = tr_t.__data__.groupby('Year').mean().reset_index()
        data = data[data['Year'].isin(range(1899, 2011))].reset_index(drop=True)
        fig, ax = climate_df.plot_full_daily_comparison(data, compare, trait, 21)
        df = climate_df.get_full_daily_comparison(data, compare, 21)
        result_plots[f'dendroclim_{trait}'] = fig
        result_tables[f'dendroclim_{trait}'] = df
    
    for trait in ['Dmean', 'Dmax', 'CWTmax', 'CWTmean', 'Dmean_ind', 'Dmax_ind', 'CWTmax_ind', 'CWTmean_ind']:
        fig, ax = tr_t.hist(trait, subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)})
        result_plots[f'hist_{trait}'] = fig
        fig, ax = tr_t.qqplot(trait, subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)})
        result_plots[f'qqplot_{trait}'] = fig
        result_tables[trait] = tr_t.describe(trait)

    for key in result_plots:
        result_plots[key].savefig(f'results/{key}.png', dpi=300)
    
    for key in result_tables:
        result_tables[key].to_excel(f'results/{key}.xlsx', index=False)


# TODO: Normal arguments
if __name__ == '__main__':
    main('app\.ipynb_checkpoints\KAZ.csv', 'app\.ipynb_checkpoints\climate_tashtyp.csv')
