import numpy as np
from scipy import stats

from src.Game import Game
from src.Grid import Grid
from src.UI.No_UI import No_UI

def confidence_interval(data, alpha=0.05):
    """
    Compute the alpha confidence interval of the mean for a given dataset.

    Parameters:
    - data (numpy array): Array of sampled values.
    - alpha (float): Significance level, default is 0.05 for a 95% confidence interval.

    Returns:
    - (float, float): The lower and upper bounds of the confidence interval.
    """
    n = len(data)  # Sample size
    mean = np.mean(data)  # Sample mean
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard error of the mean

    # Compute the t critical value
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

    # Compute the margin of error
    margin_of_error = t_crit * std_err

    # Compute the confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return ci_lower, ci_upper

def winrate(n_game, mine_percent, grid_size, bot, conf_interval: None|float=None):
    results = []
    for i in range(n_game):
        grid = Grid(grid_size, grid_size, mine_percent)
        game = Game(grid)
        result = No_UI().start(game, bot)
        results.append(result)

    if conf_interval is None:
        return np.mean(results)
    return confidence_interval(results, conf_interval)