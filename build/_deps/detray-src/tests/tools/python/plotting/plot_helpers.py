# Detray library, part of the ACTS project (R&D line)
#
# (c) 2023-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

from collections import namedtuple
import numpy as np

# -------------------------------------------------------------------------------
# Common helpers for plotting measurement data
# -------------------------------------------------------------------------------

""" Filter the data in a data frame by a given prescription """


def filter_data(data, filter=lambda df: [], variables=[]):
    data_coll = []

    # Get global data
    if len(filter(data)) == 0:
        for var in variables:
            data_coll.append(data[var].to_numpy(dtype=np.double))

    # Filtered data
    else:
        filtered = data.loc[filter]
        for var in variables:
            data_coll.append(filtered[var].to_numpy(dtype=np.double))

    return tuple(data_coll)


# -------------------------------------------------------------------------------
