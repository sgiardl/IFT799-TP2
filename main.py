"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.apriori import run_apriori
from src.dataframes import load_data, process_data, find_potential_savers, filter_rules


if __name__ == '__main__':
    data = load_data('data/bank-data.csv')

    data = process_data(data,
                        n_bins_age=5,
                        n_bin_income=5)

    data_savers = data.loc[(data['mortgage'] == 'MORTGAGE:YES') | (data['pep'] == 'PEP:YES')]
    data_savers = data_savers.drop('id', axis=1)

    data_non_savers = data.loc[(data['mortgage'] == 'MORTGAGE:NO') & (data['pep'] == 'PEP:NO')]

    rules_savers = run_apriori(data_savers,
                               min_support=0.005,
                               min_confidence=0.5,
                               min_lift=1)

    for max_length in range(min(rules_savers['rule_length']), max(rules_savers['rule_length']) + 1):
        print('*' * 50)
        print(f'Maximum Association Rules Length = {max_length}\n')

        rules_savers_filt = filter_rules(rules_savers,
                                         n_rules=10,
                                         max_length=max_length)

        find_potential_savers(data_non_savers,
                              rules_savers=rules_savers_filt,
                              max_length=max_length)

    print("Rules & clients lists have been saved to the 'csv/' & 'latex/' folders!")
