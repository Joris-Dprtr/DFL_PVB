import pandas as pd


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
        # Realistic profile
        nl_price_data = pd.read_csv('../data/NL_DA_Prices.csv', index_col='Date', parse_dates=True, dayfirst=True)
        nl_data_aggr = nl_data_aggr.merge(nl_price_data[['offtake', 'injection']], left_index=True, right_index=True)

    # calculate the cost to be a positive net load x offtake minus a negative net load x injection
    nl_data_aggr['cost'] = nl_data_aggr.apply(
        lambda row: row['net_load'] * row['injection'] if row['net_load'] < 0 else row['net_load'] * row['offtake'],
        axis=1)

    return nl_data_aggr
