"""
Utilities for extracting data from logs
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from elfpy.markets.hyperdrive import AssetIdPrefix


def read_json_to_pd(json_file):
    """
    Generic function to read json file path to pandas dataframe
    """
    # Avoids race condition if background process is writing, keep trying until it passes
    while True:
        try:
            with open(json_file, mode="r", encoding="UTF-8") as file:
                json_data = json.load(file)
            break
        except json.JSONDecodeError:
            time.sleep(0.1)
            continue
    return pd.DataFrame(json_data)


def calculate_spot_price(
    share_reserves,
    bond_reserves,
    lp_total_supply,
    maturity_timestamp=1.0,
    block_timestamp=0.0,
    position_duration=1.0,
):
    """Calculates the spot price given the pool info data"""
    # pylint: disable=too-many-arguments

    # Hard coding variables to calculate spot price
    initial_share_price = 1
    time_remaining_stretched = 0.045071688063194093
    full_term_spot_price = (
        (initial_share_price * (share_reserves / 1e18)) / ((bond_reserves / 1e18) + (lp_total_supply / 1e18))
    ) ** time_remaining_stretched

    time_left_in_years = (maturity_timestamp - block_timestamp) / position_duration

    return full_term_spot_price * time_left_in_years + 1 * (1 - time_left_in_years)


def get_combined_data(txn_data, pool_info_data):
    """
    Takes the transaction data nad pool info data and
    combines the two dataframes into a single dataframe
    """
    pool_info_data.index = pool_info_data.index.astype(int)
    # txn_data.index = txn_data["blockNumber"]
    # Combine pool info data and trans data by block number
    data = txn_data.merge(pool_info_data)

    rename_dict = {
        "event_operator": "operator",
        "event_from": "from",
        "event_to": "to",
        "event_id": "id",
        "event_prefix": "prefix",
        "event_maturity_time": "maturity_time",
        "event_value": "value",
        "bondReserves": "bond_reserves",
        "blockNumber": "block_number",
        "input_method": "trade_type",
        "longsOutstanding": "longs_outstanding",
        "longAverageMaturityTime": "longs_average_maturity_time",
        "lpTotalSupply": "lp_total_supply",
        "sharePrice": "share_price",
        "shareReserves": "share_reserves",
        "shortAverageMaturityTime": "short_average_maturity_time",
        "shortBaseVolume": "short_base_volume",
        "shortsOutstanding": "shorts_outstanding",
        "timestamp": "block_timestamp",
        "transactionHash": "transaction_hash",
        "transactionIndex": "transaction_index",
    }

    # %%
    # Filter data based on columns
    trade_data = data[list(rename_dict)]
    # Rename columns
    trade_data = trade_data.rename(columns=rename_dict)

    # TODO: Fix this -- will break if we allow multiple trades per block
    trade_data.index = trade_data["block_number"]

    # Calculate trade type and timetsamp from args.id
    def decode_prefix(row):
        # Check for nans
        if np.isnan(row):
            out = np.nan
        else:
            out = AssetIdPrefix(row).name
        return out

    trade_data["trade_enum"] = trade_data["prefix"].apply(decode_prefix)
    trade_data["timestamp"] = trade_data["block_timestamp"]
    trade_data["block_timestamp"] = trade_data["block_timestamp"].astype(int)

    return trade_data
