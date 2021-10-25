import numpy as np
import matplotlib.pyplot as plt
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

    # Removing ID column (useless information)
    data_out = data_out.drop('id', axis=1)

    # Grouping ages into ranges
    data_out = group_bins(data=data_out, col_header='age', n_bins=5, n_decimals=0)

    # Grouping income into ranges
    data_out = group_bins(data=data_out, col_header='income', n_bins=5, n_decimals=2)

    # Grouping having children as boolean YES or NO
    data_out = group_bool(data=data_out, col_header='children')

    # Appending column header to boolean values
    for column in data_out.columns:
        for i, row in data_out.iterrows():
            data_out.loc[i, column] = f'{column.upper()}:{data_out.loc[i, column]}'

    return data_out


def run_apriori(data: pd.DataFrame) -> None:
    records = []

    for i in range(len(data)):
        records.append([str(data.values[i, j]) for j in range(len(data.columns))])

    association_rules = apriori(records,
                                min_support=0.0045,
                                min_confidence=0.2,
                                min_lift=3,
                                min_length=2)
    association_results = list(association_rules)

    print(len(association_results))

    print(association_results[0])

    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])

        # second index of the inner list
        print("Support: " + str(item[1]))

        # third index of the list located at 0th
        # of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")