"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd
from apyori import apriori
from copy import deepcopy


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)

    except FileNotFoundError:
        raise FileNotFoundError(f"Please download the dataset and save it as '{filepath}' to use this script")


def save_dataframe(data: pd.DataFrame, *,
                   filename: str) -> None:
    data.to_csv(f'csv/{filename}.csv', index=False)

    with pd.option_context('max_colwidth', 1000):
        with open(f'latex/{filename}.txt', 'w') as file:
            file.write(data.to_latex(index=False))


def group_bins(data: pd.DataFrame, *,
               col_header: str,
               n_bins: int,
               n_decimals: int) -> pd.DataFrame:
    _, edges = pd.cut(data[col_header], bins=n_bins, retbins=True)
    labels = [f'({edges[i]:.{n_decimals}f}->'
              f'{edges[i + 1]:.{n_decimals}f}]' for i in range(len(edges) - 1)]

    data[col_header] = pd.cut(data[col_header], bins=n_bins, labels=labels)
    data[col_header] = data[col_header].astype(str)

    return data


def group_bool(data: pd.DataFrame, *,
               col_header: str) -> pd.DataFrame:
    for i, row in data.iterrows():
        data.loc[i, col_header] = 'YES' if data.loc[i, col_header] > 0 else 'NO'

    return data


def process_data(data: pd.DataFrame, *,
                 n_bins_age: int,
                 n_bin_income: int) -> pd.DataFrame:
    # Copying the dataframe
    data_out = deepcopy(data)

    # Grouping ages into ranges
    data_out = group_bins(data=data_out, col_header='age', n_bins=n_bins_age, n_decimals=0)

    # Grouping income into ranges
    data_out = group_bins(data=data_out, col_header='income', n_bins=n_bin_income, n_decimals=2)

    # Grouping having children as boolean YES or NO
    data_out = group_bool(data=data_out, col_header='children')

    # Appending column header to values
    for column in data_out.columns[1:]:
        for i, row in data_out.iterrows():
            data_out.loc[i, column] = f'{column.upper()}:{data_out.loc[i, column]}'

    return data_out


def run_apriori(data: pd.DataFrame, *,
                min_support: float,
                min_confidence: float,
                min_lift: float,
                verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print('Running apriori algorithm...')

    records = []

    for i in range(len(data)):
        records.append([str(data.values[i, j]) for j in range(len(data.columns))])

    association_rules = apriori(records,
                                min_support=min_support,
                                min_confidence=min_confidence,
                                min_lift=min_lift)

    columns = ['items_from', 'items_to', 'rule_length', 'support', 'confidence', 'lift']

    results_df = pd.DataFrame(columns=columns)

    association_results = list(association_rules)

    choices_base = ['MORTGAGE:YES', 'MORTGAGE:NO', 'PEP:YES', 'PEP:NO']

    choices_add = [frozenset(['MORTGAGE:YES', 'PEP:YES']),
                   frozenset(['MORTGAGE:YES', 'PEP:NO']),
                   frozenset(['MORTGAGE:NO', 'PEP:YES']),
                   frozenset(['MORTGAGE:YES']),
                   frozenset(['PEP:YES'])]

    for result in association_results:
        for order in result.ordered_statistics:
            items_base_list = list(order.items_base)
            items_add_list = list(order.items_add)
            flag_choices_base = True

            for choice_base in choices_base:
                if choice_base in items_base_list:
                    flag_choices_base = False
                    break

            if len(items_base_list) > 0 and flag_choices_base and order.items_add in choices_add:
                results_dict = {
                    columns[0]: ', '.join(items_base_list),
                    columns[1]: ', '.join(items_add_list),
                    columns[2]: len(items_base_list) + len(items_add_list),
                    columns[3]: result.support,
                    columns[4]: order.confidence,
                    columns[5]: order.lift
                }

                results_df = results_df.append(results_dict, ignore_index=True)

    results_df['score'] = results_df['confidence'] * results_df['lift']

    results_df = results_df.sort_values(by=['score', 'support'], ascending=False)

    if verbose:
        print(f'Done! Number of rules found: {len(results_df)}')

    return results_df


def filter_rules(data: pd.DataFrame, *,
                 n_rules: int,
                 max_length: int,
                 n_decimals: int = 4,
                 save: bool = True,
                 verbose: bool = True,
                 max_or_mean: str = 'max') -> pd.DataFrame:
    data = data.loc[(data['rule_length'] <= max_length)]

    data = data.head(n_rules)

    if verbose:
        print(f'Top {n_rules} Association Rules:\n{data}\n')

        if max_or_mean == 'max':
            print(f'max_support\t\t= {data["support"].max():.{n_decimals}f}')
            print(f'max_confidence\t= {data["confidence"].max():.{n_decimals}f}')
            print(f'max_lift\t\t= {data["lift"].max():.{n_decimals}f}')
            print(f'max_score\t\t= {data["score"].max():.{n_decimals}f}\n')
        elif max_or_mean == 'mean':
            print(f'mean_support\t= {data["support"].astype(float).mean():.{n_decimals}f}')
            print(f'mean_confidence\t= {data["confidence"].astype(float).mean():.{n_decimals}f}')
            print(f'mean_lift\t\t= {data["lift"].astype(float).mean():.{n_decimals}f}')
            print(f'mean_score\t\t= {data["score"].astype(float).mean():.{n_decimals}f}\n')

    for column in data.columns[3:]:
        data[column] = data[column].map(f'{{:.{n_decimals}f}}'.format)

    if save:
        save_dataframe(data, filename=f'rules-{max_length}')

    return data


def find_potential_savers(data: pd.DataFrame, *,
                          rules_savers: pd.DataFrame,
                          max_length: int,
                          save: bool = True,
                          verbose: bool = True) -> None:
    potential_savers = []

    for i, rule_saver in rules_savers.iterrows():
        rule_list = rule_saver['items_from'].split(', ')

        for j, row_non_saver in data.iterrows():
            non_saver_list = []

            for column in data.columns:
                non_saver_list.append(row_non_saver[column])

            if set(rule_list) <= set(non_saver_list):
                if row_non_saver['id'] not in potential_savers:
                    potential_savers.append(row_non_saver['id'])

    potential_savers.sort()

    if verbose:
        print(f'Potential Savers Found: {len(potential_savers)}/{len(data)}\n'
              f'List: {potential_savers}\n')

    if save:
        df = pd.DataFrame(potential_savers, columns=['col'])
        df.to_csv(f'csv/clients-{max_length}.csv', index=False, header=False)
