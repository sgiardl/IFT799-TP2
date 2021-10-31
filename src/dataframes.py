"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd
from copy import deepcopy
from string import ascii_uppercase

from src.constants import LABELS


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
            file.write(data.to_latex(index=False, escape=False))


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

    # labels_dict = {}
    labels_df = pd.DataFrame(columns=['long_label', 'short_label'])

    for column, settings in LABELS.items():
        long_labels = sorted(data_out[column].unique())

        if column == 'income':
            income_dict = {}

            for i, label in enumerate(long_labels):
                income_dict[i] = int(label.split('(')[1].split('.')[0])

            sorted_income = {k: v for k, v in sorted(income_dict.items(), key=lambda item: item[1])}

            long_labels_sorted = [long_labels[i] for i in sorted_income.keys()]
            long_labels = long_labels_sorted

        if settings['type'] == 'nominative':
            short_labels = [f'{settings["label"]}_{label.split(":")[1][0]}'
                            for label in long_labels]

        elif settings['type'] == 'categorical':
            short_labels = [f'{settings["label"]}_{list(ascii_uppercase)[i]}'
                            for i, label in enumerate(long_labels)]

        elif settings['type'] == 'boolean':
            short_labels = [f'{settings["label"]}_{1 if label.split(":")[1] == "YES" else 0}'
                            for label in long_labels]

        for i in range(len(long_labels)):
            labels_df = labels_df.append({'long_label': long_labels[i], 'short_label': short_labels[i]},
                                         ignore_index=True)

    for _, row in labels_df.iterrows():
        data_out = data_out.replace(row['long_label'], row['short_label'])

    labels_df.to_csv('csv/labels.csv', index=False)
    print("Correspondance between short & long labels has been saved to 'csv/labels.csv!'")

    return data_out


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
