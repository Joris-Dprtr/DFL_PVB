import pandas as pd


def dutch_data(path: str, aggregation: str):

    nl_data = pd.read_parquet(path)
    nl_data.loc[:, 'load'] = nl_data['grid_import'] + nl_data['solar_energy'] - nl_data['grid_export']

    # PV data
    nl_data = nl_data[['load', 'solar_energy']]

    aggregation_rules = {
        'load': 'sum',
        'solar_energy': 'sum',
    }

    nl_data_aggr = nl_data.resample(aggregation).agg(aggregation_rules)

    return nl_data_aggr
