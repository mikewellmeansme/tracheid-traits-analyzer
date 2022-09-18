import argparse
import matplotlib.pyplot as plt
import pandas as pd

from tracheid_traits import TracheidTraits
from zhutils.approximators import Polynomial
from zhutils.correlation import dropna_pearsonr
from zhutils.dataframes import DailyDataFrame
from zhutils.tracheids import Tracheids

from exponential_approximator import Exponential


plt.rcParams['font.size'] = '16'
plt.rcParams['font.family'] = 'Times New Roman'

def exponential_scatterplot(tracheid, tracheid_traits, trait_type = 'D'):
    y_min = tracheid.data.min()[f'{trait_type}mean']*0.95

    fig, axes = tracheid_traits.scatter(
        '№',
        f'{trait_type}max',
        approximator=Exponential(y_min),
        plot_kws={'color': 'black', 'linewidth': 2, 'label': f'{trait_type}max(N)'},
        scatter_kws={'color': 'white', 'edgecolor':'red', 'label': f'{trait_type}max'}, 
        subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)}
    )
    fig, axes = tracheid_traits.scatter(
        '№',
        f'{trait_type}mean',
        approximator=Exponential(y_min),
        plot_kws={'color': 'black', 'linewidth': 2, 'linestyle': '--', 'label': f'{trait_type}mean(N)'},
        axes=axes,
        scatter_kws={'color': 'white', 'edgecolor':'blue', 'label': f'{trait_type}mean'},
        xlabel='N',
        ylabel=f'{trait_type} (μm)'
    )
    for ax in axes:
        ax.legend(frameon=True)
    
    return y_min, fig, ax

def plot(tr_t, trait):
    fig, ax = tr_t.plot(trait, subplots_kws={'figsize': (7*5, 5), 'dpi':300}, plot_kws={'linewidth': 1}, mean_kws={'color': 'k', 'linewidth': 2}, std_kws={'color': 'gray', 'alpha': 0.5})
    ax.set_ylabel(f'{trait} ({"rel. units" if "ind" in trait else "μm" })')
    return fig, ax

def main(tracheid_path, climate_path, outplut_path):
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

    d_min, fig, ax = exponential_scatterplot(tr, tr_t, 'D')
    
    result_plots['D_N'] = fig

    cwt_min, fig, ax = exponential_scatterplot(tr, tr_t, 'CWT')
    
    result_plots['CWT_N'] = fig

    for trait in ['Dmean', 'Dmax', 'CWTmax', 'CWTmean']:
        df, coeffs = tr_t.apply_model('№', trait, Exponential(d_min if 'D' in trait else cwt_min))
        tr_t.add_trait(f'{trait}_model', df)
        loc_df = tr_t.get_traits([trait, f'{trait}_model'])
        loc_df[f'{trait}_ind'] = loc_df[trait] / loc_df[f'{trait}_model']
        tr_t.add_trait(f'{trait}_ind', loc_df)
        coeffs['Coef'] = ['Y_min', 'Y_as', 'a']
        result_tables[f'coeffs_{trait}'] = pd.DataFrame(coeffs)
    
    result_tables['TracheidTraits'] = tr_t.get_traits()
    
    traits = ['Dmean', 'Dmax', 'CWTmax', 'CWTmean', 'Dmean_ind', 'Dmax_ind', 'CWTmax_ind', 'CWTmean_ind', 'TRW']
    
    for trait in traits:
        def compare(df, index):
            r, p = dropna_pearsonr(df[index], df[trait])
            return r, p

        data = tr_t.__data__.groupby('Year').mean().reset_index()
        data = data[data['Year'].isin(range(1899, 2011))].reset_index(drop=True)
        fig, ax = climate_df.plot_full_daily_comparison(data, compare, trait, 21)
        df = climate_df.get_full_daily_comparison(data, compare, 21)
        result_plots[f'dendroclim_plot_{trait}'] = fig
        result_tables[f'dendroclim_table_{trait}'] = df
    
    for trait in traits:
        fig, ax = tr_t.hist(trait, subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)})
        result_plots[f'hist_{trait}'] = fig
        fig, ax = tr_t.qqplot(trait, subplots_kws={'sharex': 'all', 'sharey': 'all', 'figsize': (5*5, 5)})
        result_plots[f'qqplot_{trait}'] = fig
        result_tables[f'table_{trait}'] = tr_t.describe(trait)
    
    plt.rcParams['font.size'] = '18'

    for trait in traits:
        fig, ax = plot(tr_t, trait)
        result_plots[f'plot_{trait}'] = fig

    for key in result_plots:
        result_plots[key].savefig(f'{outplut_path}/{key}.png', dpi=300)
    
    for key in result_tables:
        result_tables[key].to_excel(f'{outplut_path}/{key}.xlsx', index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="""
        Generates tables and graphs describing the distribution of tracheid traits,
        their change over time,
        correlation with daily climate data,
        and approximation (linear for TRW and exponential for D and CWT) with line coefficients.
    """)

    parser.add_argument("tracheid_path", help="Path to csv tracheid data in Tracheids format.")
    parser.add_argument("climate_path", help="Path to csv file with daily climate data in DailyDataFrame format.")
    parser.add_argument("output_path", help="Path to save obtained plots and tables.")

    args = parser.parse_args()

    main(args.tracheid_path, args.climate_path, args.output_path)
