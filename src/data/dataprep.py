import pandas as pd
from pathlib import Path

def _assign_value(time, night, day):
    if time < pd.Timestamp("08:00:00").time() or time >= pd.Timestamp("19:00:00").time():
        return night
    else:
        return day


def dutch_data(path: str, aggregation: str, price='Realistic'):
    nl_data = pd.read_parquet(path)
    nl_data.loc[:, 'load'] = nl_data['grid_import'] + nl_data['solar_energy'] - nl_data['grid_export']

    # PV data
    nl_data = nl_data[['load', 'solar_energy']]

    aggregation_rules = {
        'load': 'sum',
        'solar_energy': 'sum',
    }

    nl_data_aggr = nl_data.resample(aggregation).agg(aggregation_rules)

    # Include net load for cost calculations
    nl_data_aggr['net_load'] = nl_data_aggr['load'] - nl_data_aggr['solar_energy']
    # Import a price profile and merge it with the consumption data

    if price == 'Simple':
        # Simple profile
        nl_data_aggr.loc[:, 'offtake'] = nl_data_aggr.index.to_series().apply(
            lambda x: _assign_value(x.time(), 0.2, 0.3))
        nl_data_aggr.loc[:, 'injection'] = nl_data_aggr.index.to_series().apply(
            lambda x: _assign_value(x.time(), 0.05, 0.01))
    elif price == 'Realistic':
        # Resolve the path to the provided file
        building_path = Path(path).resolve()

        # Try to find the parent 'data' folder
        parts = building_path.parts
        if 'data' in parts:
            data_index = parts.index('data')
            base_data_path = Path(*parts[:data_index + 1])
        else:
            raise ValueError("Provided path must include a 'data' directory.")

        # Path to price data
        price_path = base_data_path / 'NL_DA_Prices.csv'
        nl_price_data = pd.read_csv(price_path, index_col='Date', parse_dates=True, dayfirst=True)

        # Merge price data
        nl_data_aggr = nl_data_aggr.merge(nl_price_data[['offtake', 'injection']], left_index=True, right_index=True)

        # Calculate cost
        nl_data_aggr['cost'] = nl_data_aggr.apply(
            lambda row: row['net_load'] * row['injection'] if row['net_load'] < 0 else row['net_load'] * row['offtake'],
        axis=1)

    return nl_data_aggr
