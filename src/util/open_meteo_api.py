import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from scipy.stats import skewnorm
from scipy.ndimage import gaussian_filter1d


def _apply_random_time_shifts(series, shift_prob=0.1, max_shift=1):
    """
    Shift values forward/backward in time with given probability.

    Parameters:
    - series: input array
    - shift_prob: probability of shifting each point
    - max_shift: max steps to shift (positive or negative)

    Returns:
    - shifted array (same length, interpolated)
    """
    shifted = np.full_like(series, np.nan)
    for i in range(len(series)):
        if np.random.rand() < shift_prob:
            shift = np.random.randint(-max_shift, max_shift + 1)
            new_i = np.clip(i + shift, 0, len(series) - 1)
            shifted[new_i] = series[i]
        else:
            shifted[i] = series[i]

    # Fill in NaNs by interpolation
    mask = np.isnan(shifted)
    shifted[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), shifted[~mask])

    return shifted

def simulate_forecast(actual, noise_scale=0.1, skewness=-2, smooth_param=5):
    """
    Simulate forecast using a skewed noise distribution and then smooth the predictions.

    Parameters:
    - actual: Series or array of actual values
    - noise_scale: scale (std dev) of noise applied as % of actual
    - skewness: negative values will skew the noise left (underestimation)
    - smooth_method: method to smooth predictions ('rolling', 'ema', 'gaussian')
    - smooth_param: parameter for smoothing method (window size, span, sigma)

    Returns:
    - forecasted: Smoothed series of simulated forecasts
    """
    # Generate skewed noise from skew-normal distribution
    noise = skewnorm.rvs(a=skewness, size=len(actual)) * noise_scale

    # Apply noise
    forecasted = actual * (1 + noise)
    forecasted = forecasted.clip(lower=min(actual))
    forecasted = forecasted.clip(upper=np.sort(actual)[-10])
    forecasted = _apply_random_time_shifts(forecasted, shift_prob=noise_scale/10, max_shift=5)

    # Apply smoothing based on selected method
    forecasted = gaussian_filter1d(forecasted, sigma=smooth_param)

    return forecasted


class Open_meteo:

    def __init__(self,
                 latitude,
                 longitude,
                 start_date,
                 end_date):
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "weather_code", "cloud_cover", "direct_radiation",
                       "diffuse_radiation"],
            "models": "best_match"
        }
        self.variables = ['date'] + ['relative_humidity_2m', 'diffuse_radiation', 'direct_radiation', 'temperature_2m',
                                     'weather_code', 'cloud_cover']

    def get_open_meteo_hourly(self):
        # Set up the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        open_meteo = openmeteo_requests.Client(session=retry_session)

        responses = open_meteo.weather_api(self.url, params=self.params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(2).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
        hourly_direct_radiation = hourly.Variables(4).ValuesAsNumpy()
        hourly_diffuse_radiation = hourly.Variables(5).ValuesAsNumpy()
        date = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left")

        hourly_data = {"date": date,
                       "temperature_2m": hourly_temperature_2m,
                       "relative_humidity_2m": hourly_relative_humidity_2m,
                       "weather_code": hourly_weather_code,
                       "cloud_cover": hourly_cloud_cover,
                       "direct_radiation": hourly_direct_radiation,
                       "diffuse_radiation": hourly_diffuse_radiation}

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframe = hourly_dataframe[self.variables]

        return hourly_dataframe
