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
    return pd.read_csv(filepath)


def group_bins(data: pd.DataFrame,
               col_header: str,
               n_bins: int,
               n_decimals: int) -> pd.DataFrame:
    _, edges = pd.cut(data[col_header], bins=n_bins, retbins=True)
    labels = [f'({edges[i]:.{n_decimals}f}, '
              f'{edges[i + 1]:.{n_decimals}f}]' for i in range(len(edges) - 1)]

    data[col_header] = pd.cut(data[col_header], bins=n_bins, labels=labels)
    data[col_header] = data[col_header].astype(str)

    return data


def group_bool(data: pd.DataFrame,
               col_header: str) -> pd.DataFrame:
    for i, row in data.iterrows():
        data.loc[i, col_header] = 'YES' if data.loc[i, col_header] > 0 else 'NO'

    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # Copying the dataframe
    data_out = deepcopy(data)

    # Grouping ages into ranges
    data_out = group_bins(data=data_out, col_header='age', n_bins=5, n_decimals=0)

    # Grouping income into ranges
    data_out = group_bins(data=data_out, col_header='income', n_bins=5, n_decimals=2)

    # Grouping having children as boolean YES or NO
    data_out = group_bool(data=data_out, col_header='children')

    # Appending column header to values
    for column in data_out.columns:
        for i, row in data_out.iterrows():
            data_out.loc[i, column] = f'{column.upper()}:{data_out.loc[i, column]}'

    return data_out


def run_apriori(data: pd.DataFrame,
                n_rules: int,
                min_support: float,
                min_confidence: float,
                min_lift: float,
                max_length: int) -> pd.DataFrame:
    # Removing ID column (useless information)
    data = data.drop('id', axis=1)

    records = []

    for i in range(len(data)):
        records.append([str(data.values[i, j]) for j in range(len(data.columns))])

    association_rules = apriori(records,
                                min_support=min_support,
                                min_confidence=min_confidence,
                                min_lift=min_lift,
                                max_length=max_length)

    columns = ['items_from', 'items_to', 'support', 'confidence', 'lift']

    results_df = pd.DataFrame(columns=columns)

    association_results = list(association_rules)

    choices = [frozenset(['MORTGAGE:YES', 'PEP:YES']),
               frozenset(['MORTGAGE:YES']),
               frozenset(['PEP:YES'])]

    for result in association_results:
        for order in result.ordered_statistics:
            if len(order.items_base) > 0 and order.items_add in choices:
                results_dict = {
                    columns[0]: ', '.join(list(order.items_base)),
                    columns[1]: ', '.join(list(order.items_add)),
                    columns[2]: result.support,
                    columns[3]: order.confidence,
                    columns[4]: order.lift
                }

                results_df = results_df.append(results_dict, ignore_index=True)

    # Combining confidence and lift into 1 score to choose the top N association rules
    results_df['score'] = results_df['confidence'] / results_df['confidence'].max() / 2 + \
                          results_df['lift'] / results_df['lift'].max() / 2

    results_df = results_df.sort_values(by=['score'], ascending=False).head(n_rules)

    results_df = results_df.drop('score', axis=1)

    return results_df


def find_potential_savers(data: pd.DataFrame,
                          rules_savers: pd.DataFrame) -> None:
    potential_savers = []

    for i, rule_saver in rules_savers.iterrows():
        rule_list = rule_saver['items_from'].split(', ')

        for j, row_non_saver in data.iterrows():
            non_saver_list = []

            for column in data.columns:
                non_saver_list.append(row_non_saver[column])

            if set(rule_list) <= set(non_saver_list) and \
                    row_non_saver['id'] not in potential_savers:
                potential_savers.append(row_non_saver['id'])

    print(f'Potential Savers List:\n{potential_savers}')
