"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd

from src.apriori import run_apriori
from src.dataframes import load_data, process_data, filter_rules, save_dataframe
from src.utils import get_combinations_of_two

if __name__ == '__main__':
    data = load_data('data/bank-data.csv')

    columns = ['n_bins_age', 'n_bins_income', 'mean_support', 'mean_confidence', 'mean_lift', 'mean_score']
    results_df = pd.DataFrame(columns=columns)

    runs_tuples = get_combinations_of_two(n_min=2, n_max=10, include_rev=True)

    for i, (n_bins_age, n_bins_income) in enumerate(runs_tuples, start=1):
        print('*' * 50)
        print(f'Run {i}/{len(runs_tuples)}')
        print(f'{n_bins_age = }, {n_bins_income = }')

        data_processed = process_data(data,
                                      n_bins_age=n_bins_age,
                                      n_bin_income=n_bins_income)

        data_savers = data_processed.loc[(data_processed['mortgage'] == 'MORTGAGE:YES') | (data_processed['pep'] == 'PEP:YES')]
        data_savers = data_savers.drop('id', axis=1)

        data_non_savers = data_processed.loc[(data_processed['mortgage'] == 'MORTGAGE:NO') & (data_processed['pep'] == 'PEP:NO')]

        rules_savers = run_apriori(data_savers,
                                   min_support=0.005,
                                   min_confidence=0.5,
                                   min_lift=1,
                                   verbose=False)

        rules_savers_filt = filter_rules(rules_savers,
                                         n_rules=10,
                                         max_length=max(rules_savers['rule_length']),
                                         save=False,
                                         verbose=False)

        mean_support = rules_savers_filt["support"].astype(float).mean()
        mean_confidence = rules_savers_filt["confidence"].astype(float).mean()
        mean_lift = rules_savers_filt["lift"].astype(float).mean()
        mean_score = rules_savers_filt["score"].astype(float).mean()

        print(f'mean_support\t= {mean_support:.4f}')
        print(f'mean_confidence\t= {mean_confidence:.4f}')
        print(f'mean_lift\t\t= {mean_lift:.4f}')
        print(f'mean_score\t\t= {mean_score:.4f}\n')

        results_dict = {
            columns[0]: n_bins_age,
            columns[1]: n_bins_income,
            columns[2]: mean_support,
            columns[3]: mean_confidence,
            columns[4]: mean_lift,
            columns[5]: mean_score
        }

        results_df = results_df.append(results_dict, ignore_index=True)

    save_dataframe(results_df, filename='n_bins')
    print("n_bins optimization results have been saved to the 'csv/' & 'latex/' folders!")
