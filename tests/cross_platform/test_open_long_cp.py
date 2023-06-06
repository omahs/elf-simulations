"""
Test initialization of markets.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import ape

# elfpy core repo
from conftest import TestCaseWithHyperdriveFixture
from elfpy.math.fixed_point import FixedPoint
from tests.cross_platform import utils

if TYPE_CHECKING:
    from ape.contracts.base import ContractInstance


class TestOpenLongCrossPlatform(TestCaseWithHyperdriveFixture):
    """Test case for initializing the market"""

    APPROX_EQ = FixedPoint(scaled_value=10)

    def test_market_open_long(self):
        """Verify both markets initialized correctly."""

        self.inititalize()

        fx = self.fixture  # pylint: disable=invalid-name
        deployer = fx.deployer
        config = fx.config
        position_duration_seconds = config.position_duration_seconds
        checkpoint_duration_days = int(config.checkpoint_duration_seconds / 60 / 60 / 24)

        market_state_sol_before = utils.get_simulation_market_state_from_contract(
            hyperdrive_data_contract=fx.contracts.hyperdrive_data_contract,
            agent_address=deployer.address,
            position_duration_seconds=FixedPoint(position_duration_seconds),
            checkpoint_duration_days=FixedPoint(checkpoint_duration_days),
            variable_apr=config.initial_apr,
            config=config,
        )

        with ape.accounts.use_sender(fx.agents.solidity.alice):
            alice_balance = fx.contracts.base_erc20.balanceOf()
            print(f"{alice_balance=}")
            fx.contracts.hyperdrive_contract.openLong()
