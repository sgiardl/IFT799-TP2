"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd
from apyori import apriori


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

    # choices_base = ['MORTGAGE:YES', 'MORTGAGE:NO', 'PEP:YES', 'PEP:NO']
    #
    # choices_add = [frozenset(['MORTGAGE:YES', 'PEP:YES']),
    #                frozenset(['MORTGAGE:YES', 'PEP:NO']),
    #                frozenset(['MORTGAGE:NO', 'PEP:YES']),
    #                frozenset(['MORTGAGE:YES']),
    #                frozenset(['PEP:YES'])]

    choices_base = ['MG_1', 'MG_0', 'P_1', 'P_0']

    choices_add = [frozenset(['MG_1', 'P_1']),
                   frozenset(['MG_1', 'P_0']),
                   frozenset(['MG_0', 'P_1']),
                   frozenset(['MG_1']),
                   frozenset(['P_1'])]

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
