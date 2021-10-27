"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.functions import load_data, process_data, run_apriori, find_potential_savers

"""
Investigations:

- Doing apriori on data instead of data_savers?
- Sorting association rules first on score combining (confidence + lift) and second on support,
  even though it states to use (confidence + lift) to choose rules?
- Changing bins for age + income?
- Changing children to bins instead of boolean?

"""

if __name__ == '__main__':
    data = load_data('data/bank-data.csv')

    data = process_data(data,
                        n_bins_age=5,
                        n_bin_income=5)

    data_savers = data.loc[(data['mortgage'] == 'MORTGAGE:YES') | (data['pep'] == 'PEP:YES')]
    data_savers = data_savers.drop('id', axis=1)

    data_non_savers = data.loc[(data['mortgage'] == 'MORTGAGE:NO') & (data['pep'] == 'PEP:NO')]

    rules_dict = {}

    max_rules_length = len(data_savers.columns) - 2

    for max_length in range(3, max_rules_length + 1):
        print('*' * 50)
        print(f'Maximum Association Rules Length = {max_length}\n')

        rules_savers = run_apriori(data_savers,
                                   n_rules=10,
                                   min_support=0.005,
                                   min_confidence=0.5,
                                   min_lift=1,
                                   max_length=max_length)

        rules_dict[f'{max_length = }'] = rules_savers

        find_potential_savers(data_non_savers,
                              rules_savers=rules_savers)
