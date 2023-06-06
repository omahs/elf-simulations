"""Helper functions for post-processing simulation outputs"""
from __future__ import annotations  # types will be strings by default in 3.11

import pandas as pd

import elfpy.simulators as simulators
from elfpy.markets.hyperdrive import hyperdrive_actions, HyperdrivePricingModel
from elfpy.math import FixedPoint

# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false


def aggregate_simulation_state(combined_trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the trade details from the simulation state into a representation of market & wallet states per trade"""
    # one pnl calculation per trade
    pnl_states = []
    # one market state per trade
    market_states = [combined_trades_df.loc[0, "market_init"]]
    # wallet states are organized by agent first, then by trade
    wallet_states = []
    for agent_init in combined_trades_df.loc[0, "agent_init"]:
        wallet_states.append([agent_init])
    for trade_index, trade in combined_trades_df.iterrows():
        # get deltas
        market_deltas = trade.market_deltas
        agent_address = trade.agent_address
        agent_deltas = trade.agent_deltas
        # compute new market state from previous state
        new_market_state = market_states[trade_index].copy().apply_delta(market_deltas)
        new_wallet_state = wallet_states[agent_address][trade_index].copy().apply_delta(agent_deltas)
        # add entry to the list
        market_states.append(new_market_state)
        wallet_states[agent_address].append(new_wallet_state)
        ### PNL calculations
        lp_token_value = FixedPoint(0)
        # proceed further only if the agent has LP tokens and avoid divide by zero
        if new_wallet_state.lp_tokens > FixedPoint(0) and new_market_state.lp_total_supply > FixedPoint(0):
            share_of_pool = new_wallet_state.lp_tokens / new_market_state.lp_total_supply
            pool_value = (
                new_market_state.bond_reserves * trade.spot_price  # in base
                + new_market_state.share_reserves * new_market_state.share_price  # in base
            )
            lp_token_value = pool_value * share_of_pool  # in base
        share_reserves = new_market_state.share_reserves
        # compute long values in units of base
        longs_value = FixedPoint(0)
        longs_value_no_mock = FixedPoint(0)
        for mint_time, long in new_wallet_state.longs.items():
            if long.balance > FixedPoint(0) and share_reserves:
                balance = hyperdrive_actions.calc_close_long(
                    bond_amount=long.balance,
                    market_state=new_market_state,
                    position_duration=trade.position_duration,
                    pricing_model=HyperdrivePricingModel(),
                    block_time=trade.time,
                    mint_time=mint_time,
                    is_trade=True,
                )[1].balance.amount
            else:
                balance = FixedPoint(0)
            longs_value += balance
            longs_value_no_mock += long.balance * trade.spot_price
        # compute short values in units of base
        shorts_value = FixedPoint(0)
        shorts_value_no_mock = FixedPoint(0)
        for mint_time, short in new_wallet_state.shorts.items():
            balance = FixedPoint(0)
            if (
                short.balance > FixedPoint(0)
                and share_reserves > FixedPoint(0)
                and new_market_state.bond_reserves - new_market_state.bond_buffer > short.balance
            ):
                balance = hyperdrive_actions.calc_close_short(
                    bond_amount=short.balance,
                    market_state=new_market_state,
                    position_duration=trade.position_duration,
                    pricing_model=HyperdrivePricingModel(),
                    block_time=trade.time,
                    mint_time=mint_time,
                    open_share_price=short.open_share_price,
                )[1].balance.amount
            shorts_value += balance
            base_no_mock = short.balance * (FixedPoint("1.0") - trade.spot_price)
            shorts_value_no_mock += base_no_mock
        pnl_states.append(
            {
                f"agent_{new_wallet_state.address}_base": new_wallet_state.balance.amount,
                f"agent_{new_wallet_state.address}_lp_tokens": lp_token_value,
                f"agent_{new_wallet_state.address}_num_longs": FixedPoint(len(new_wallet_state.longs)),
                f"agent_{new_wallet_state.address}_num_shorts": FixedPoint(len(new_wallet_state.shorts)),
                f"agent_{new_wallet_state.address}_total_longs": longs_value,
                f"agent_{new_wallet_state.address}_total_shorts": shorts_value,
                f"agent_{new_wallet_state.address}_total_longs_no_mock": longs_value_no_mock,
                f"agent_{new_wallet_state.address}_total_shorts_no_mock": shorts_value_no_mock,
            }
        )
    return market_states, wallet_states, pnl_states


def aggregate_trade_data(trades: pd.DataFrame) -> pd.DataFrame:
    r"""Aggregate trades dataframe by computing means

    Arguments
    ----------
    trades : DataFrame
        Pandas dataframe containing the simulation_state keys as columns, as well as some computed columns

    Returns
    -------
    trades_agg : DataFrame
        aggregated dataframe that keeps the model_name and day columns
        and computes the mean over spot price
    """
    ### STATS AGGREGATED BY SIM AND DAY ###
    # aggregates by two dimensions:
    # 1. model_name (directly output from pricing_model class)
    # 2. day
    keep_columns = [
        "model_name",
        "day",
    ]
    trades_agg = trades.groupby(keep_columns).agg(
        {
            "spot_price": ["mean"],
            "delta_base_abs": ["sum"],
        }
    )
    trades_agg.columns = ["_".join(col).strip() for col in trades_agg.columns.values]
    trades_agg = trades_agg.reset_index()
    return trades_agg


def get_simulation_state_df(simulator: simulators.Simulator) -> pd.DataFrame:
    r"""Converts the simulator output dictionary to a pandas dataframe

    Arguments
    ----------
    simulator : Simulator
        Simulator object that holds the simulation_state

    Returns
    -------
    trades : DataFrame
        Pandas dataframe containing the simulation_state keys as columns, as well as some computed columns

    .. todo::
        Using the new sim state:
        # def get_simulation_state_df(simulator: Simulator) -> pd.DataFrame:
        #      return simulator.simulation_state.combined_dataframe

        Also, converting from FixedPoint (which gets cast to "object") to real types needs to be fixed
    """
    # construct dataframe from simulation dict
    sim_dict = simulator.simulation_state.__dict__
    if "frozen" in sim_dict:
        del sim_dict["frozen"]
    if "no_new_attribs" in sim_dict:
        del sim_dict["no_new_attribs"]
    trades_df = pd.DataFrame.from_dict(sim_dict)
    string_columns = [
        "model_name",
    ]
    int_columns = [
        "checkpoint_duration_days",
        "run_number",
        "day",
        "block_number",
        "daily_block_number",
        "trade_number",
    ]
    float_columns = [
        "base_buffer",
        "bond_buffer",
        "bond_reserves",
        "checkpoint_duration",
        "current_time",
        "current_variable_apr",
        "curve_fee_multiple",
        "flat_fee_multiple",
        "fixed_apr",
        "gov_fees_accrued",
        "governance_fee_multiple",
        "init_share_price",
        "long_average_maturity_time",
        "long_base_volume",
        "longs_outstanding",
        "lp_total_supply",
        "run_number",
        "share_reserves",
        "share_price",
        "short_average_maturity_time",
        "short_base_volume",
        "shorts_outstanding",
        "spot_price",
        "time_step_size",
        "total_supply_withdraw_shares",
        "variable_apr",
        "withdraw_capital",
        "withdraw_interest",
        "withdraw_shares_ready_to_withdraw",
    ]
    trades_df[float_columns] = trades_df[float_columns].astype(float)
    trades_df[int_columns] = trades_df[int_columns].astype(int)
    trades_df[string_columns] = trades_df[string_columns].astype(str)
    for col in list(trades_df):
        if col.startswith("agent"):  # type: ignore
            trades_df[col] = trades_df[col].astype(float)
    return trades_df


def compute_derived_variables(simulator: simulators.Simulator) -> pd.DataFrame:
    r"""Converts the simulator output dictionary to a pandas dataframe and computes derived variables

    Argument
    ----------
    simulator : Simulator
        Simulator object that holds the simulation_state

    Returns
    -------
    trades : DataFrame
        Pandas dataframe containing the simulation_state keys as columns, as well as some computed columns
    """
    trades_df = get_simulation_state_df(simulator)
    # calculate changes in reserves, corresponding to latest trade
    trades_df["delta_shares"] = trades_df.share_reserves.diff()
    trades_df["delta_base"] = trades_df.share_reserves.diff() * trades_df.share_price
    trades_df["delta_bonds"] = trades_df.bond_reserves.diff()
    # same thing but with absolute values for plotting
    trades_df["delta_shares_abs"] = trades_df.delta_shares.abs()
    trades_df["delta_base_abs"] = trades_df.delta_base.abs()
    trades_df["delta_bonds_abs"] = trades_df.delta_bonds.abs()
    # calculate derived variables across runs
    trades_df["fixed_apr_percent"] = trades_df.fixed_apr * 100
    trades_df["variable_apr_percent"] = trades_df.variable_apr * 100
    share_liquidity_usd = trades_df.share_reserves * trades_df.share_price
    bond_liquidity_usd = trades_df.bond_reserves * trades_df.share_price * trades_df.spot_price
    trades_df["total_liquidity_usd"] = share_liquidity_usd + bond_liquidity_usd
    # calculate percent change in spot price since the first spot price (after first trade)
    trades_df["price_total_return"] = (
        trades_df.loc[:, "spot_price"] / trades_df.loc[0, "spot_price"] - 1  # type: ignore
    )
    trades_df["price_total_return_percent"] = trades_df.price_total_return * 100
    # rescale price_total_return to equal init_share_price for the first value, for comparison
    trades_df["price_total_return_scaled_to_share_price"] = (
        trades_df.price_total_return + 1
    ) * trades_df.init_share_price  # this is APR (does not include compounding)
    # compute the total return from share price
    trades_df["share_price_total_return"] = 0
    for run in trades_df.run_number.unique():
        trades_df.loc[trades_df.run_number == run, "share_price_total_return"] = (
            trades_df.loc[trades_df.run_number == run, "share_price"]
            / trades_df.loc[trades_df.run_number == run, "share_price"].iloc[0]  # type: ignore
            - 1
        )
    trades_df["share_price_total_return_percent"] = trades_df.share_price_total_return * 100
    # compute rescaled returns to common annualized metric
    scale = 365 / (trades_df["day"] + 1)
    trades_df["price_total_return_percent_annualized"] = scale * trades_df["price_total_return_percent"]
    trades_df["share_price_total_return_percent_annualized"] = scale * trades_df["share_price_total_return_percent"]
    # create explicit column that increments per trade
    add_pnl_columns(trades_df)
    trades_df = trades_df.reset_index()
    return trades_df


def add_pnl_columns(trades_df: pd.DataFrame) -> None:
    """Adds Profit and Loss Column for every agent to the dataframe that is passed in"""
    num_agents = len([col for col in trades_df if str(col).startswith("agent") and str(col).endswith("base")])
    for agent_id in range(num_agents):
        wallet_values_in_base = [
            f"agent_{agent_id}_base",
            f"agent_{agent_id}_lp_tokens",
            f"agent_{agent_id}_total_longs",
            f"agent_{agent_id}_total_shorts",
        ]
        wallet_values_in_base_no_mock = [
            f"agent_{agent_id}_base",
            f"agent_{agent_id}_lp_tokens",
            f"agent_{agent_id}_total_longs_no_mock",
            f"agent_{agent_id}_total_shorts_no_mock",
        ]
        trades_df[f"agent_{agent_id}_pnl"] = trades_df[wallet_values_in_base].astype(float).sum(axis=1)
        trades_df[f"agent_{agent_id}_pnl_no_mock"] = trades_df[wallet_values_in_base_no_mock].astype(float).sum(axis=1)
