"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.functions import load_data, process_data, run_apriori, find_potential_savers

if __name__ == '__main__':
    data = load_data('data/bank-data.csv')

    data = process_data(data)

    data_savers = data.loc[(data['mortgage'] == 'MORTGAGE:YES') | (data['pep'] == 'PEP:YES')]
    data_non_savers = data.loc[(data['mortgage'] == 'MORTGAGE:NO') & (data['pep'] == 'PEP:NO')]

    rules_savers = run_apriori(data_savers,
                               n_rules=10,
                               min_support=0.001,  # 0.01
                               min_confidence=0.01,  # 0.5
                               min_lift=1,  # 20
                               max_length=5)  # 6 takes a long time but gives good results

    find_potential_savers(data_non_savers, rules_savers)
