from tsfb.base.evaluation.strategy.fixed_forecast import FixedForecast
from tsfb.base.evaluation.strategy.rolling_forecast import RollingForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
}
