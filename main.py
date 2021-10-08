"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('data/bank-data.csv')

    print(data)
