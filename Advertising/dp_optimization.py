import math
import pandas as pd

import numpy as np

# minus infinity value
m_inf = -math.inf


def knapsack_optimizer(table, budget):

    rows = len(table)
    cols = len(table[0])

    # set negative values to -inf
    for row in range(rows):
        for col in range(cols):
            if table[row][col] < 0:
                table[row][col] = m_inf

    # optimization table
    opt_table = [[] for row in range(rows)]

    # pointer matrix
    opt_indexes = [[] for row in range(rows - 1)]

    # copy the value of the first sub-campaign in the optimization table
    for col in range(cols):
        opt_table[0].append(table[0][col])

    # optimization algorithm
    for row in range(1, rows):
        for col in range(cols):
            temp = []

            for col2 in range(col + 1):
                temp.append(table[row][col2] + opt_table[row - 1][col - col2])

            max_value = max(temp)

            # update tables of values and pointers
            opt_table[row].append(max_value)
            opt_indexes[row - 1].append(temp.index(max_value))

    # optimal cumulative number of clicks
    optimal_clicks = opt_table[rows - 1]
    for i in range(cols):
        optimal_clicks[i] = optimal_clicks[i] - budget[i]
    opt_value = max(optimal_clicks)

    # pointer to the optimal budget column
    opt_col = optimal_clicks.index(opt_value)

    # list of budget-pointers for each sub-campaign
    assignments = [0 for r in range(rows)]

    for row in range(rows - 1, 0, -1):
        # index of the optimal sub-campaign budget
        subc_col = opt_indexes[row - 1][opt_col]

        assignments[row] = subc_col
        opt_col -= subc_col

    # assign the index of the first sub-campaign
    assignments[0] = opt_col

    # return list(enumerate(assignments))
    return assignments


def get_knapsack_values(table, assignments):
    values = []
    for row in range(len(assignments)):
        values.append(table[row][assignments[row]])
    return values


def get_dataframe(table, assignments, columns):

    df = pd.DataFrame(data=table,
                      index=pd.Index(["C" + str(i) for i in range(len(table))]),
                      columns=columns
                      )
    df.insert(len(table[0]), 'Budget', assignments)
    return df
