"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from src.functions import load_data, process_data, run_apriori

if __name__ == '__main__':
    data = load_data('data/bank-data.csv')

    data = process_data(data)

    run_apriori(data)
