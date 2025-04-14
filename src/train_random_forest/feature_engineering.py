import pandas as pd
import numpy as np


def delta_date_feature(dates):
    """
    Given a 2D array containing dates (in any format recognized by pd.to_datetime),
    this function returns the delta in days between each date and the most recent date in its column.

    Parameters:
        dates (array-like): 2D array of date strings or datetime objects.

    Returns:
        numpy.ndarray: Array of the same shape with number of days from the most recent date.
    """
    # Convert input to pandas datetime
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)

    # Calculate delta from max date
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()
