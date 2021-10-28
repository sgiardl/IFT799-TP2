"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.functions import load_data, process_data, run_apriori, filter_rules


if __name__ == '__main__':
    data = load_data('data/bank-data.csv')

    for n_bins in range(2, 11):
        print('*' * 50)
        print(f'{n_bins = }')

        data_processed = process_data(data,
                                      n_bins_age=n_bins,
                                      n_bin_income=n_bins)

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
                                         save=False)
