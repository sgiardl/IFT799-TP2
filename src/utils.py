"""
IFT799 - Science des données
TP2
Automne 2021
Olivier Lefebvre
Simon Giard-Leroux
"""

from itertools import combinations


def get_combinations_of_two(n_min: int,
                            n_max: int, *,
                            include_rev: bool) -> list:
    n_list = list(range(n_min, n_max + 1))

    combs_of_two = []

    for comb in combinations(n_list, 2):
        combs_of_two.append(tuple(comb))

        if include_rev:
            combs_of_two.append(tuple(comb))

    if include_rev:
        for i in range(len(combs_of_two)):
            if i % 2 != 0:
                combs_of_two[i] = combs_of_two[i][::-1]

    for n in n_list:
        combs_of_two.append((n, n))

    combs_of_two.sort()

    return combs_of_two
