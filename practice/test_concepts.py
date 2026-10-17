"""One small test per concept, written while walking through generate.py.

Each test has a WRONG number in it on purpose. Fix the number so the test passes.
Rule: write your prediction on paper BEFORE you run pytest.
"""

import numpy as np


def test_grid_row_and_column_numbers() -> None:
    # A table with 2 rows and 5 columns. y holds each box's row number,
    # x holds each box's column number.
    y, x = np.mgrid[0:2, 0:5]

    assert y.shape == (5, 2)   # TODO: (rows, columns) of the table. Fix it.
    assert y[1, 3] == 0        # TODO: the box in row 1, column 3 -> what row number is stored?
    assert x[1, 3] == 0        # TODO: same box -> what column number is stored?
