"""CRUD tests for WalletInfo"""
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from elfpy.data import postgres
from elfpy.data.db_schema import Base, WalletInfo

engine = create_engine("sqlite:///:memory:")  # in-memory SQLite database for testing
Session = sessionmaker(bind=engine)

# fixture arguments in test function have to be the same as the fixture name
# pylint: disable=redefined-outer-name

# Explicitly testing protected access function, i.e., _get_latest_block_number_wallet_info(session)
# pylint: disable=protected-access


@pytest.fixture(scope="function")
def session():
    """Session fixture for tests"""
    Base.metadata.create_all(engine)  # create tables
    session_ = Session()
    yield session_
    session_.close()
    Base.metadata.drop_all(engine)  # drop tables


class TestWalletInfoTable:
    """CRUD tests for WalletInfo table"""

    def test_create_wallet_info(self, session):
        """Create and entry"""
        # Note: this test is using inmemory sqlite, which doesn't seem to support
        # autoincrementing ids without init, whereas postgres does this with no issues
        # Hence, we explicitly add id here
        wallet_info = WalletInfo(blockNumber=1, tokenValue=3.2)  # add your other columns here...
        session.add(wallet_info)
        session.commit()

        retrieved_wallet_info = session.query(WalletInfo).filter_by(blockNumber=1).first()
        assert retrieved_wallet_info is not None
        # toekValue retreieved from postgres is in Decimal, cast to float
        assert float(retrieved_wallet_info.tokenValue) == 3.2

    def test_update_wallet_info(self, session):
        """Update an entry"""
        wallet_info = WalletInfo(blockNumber=1, tokenValue=3.2)
        session.add(wallet_info)
        session.commit()

        wallet_info.tokenValue = 5.0
        session.commit()

        updated_wallet_info = session.query(WalletInfo).filter_by(blockNumber=1).first()
        # tokenValue retreieved from postgres is in Decimal, cast to float
        assert float(updated_wallet_info.tokenValue) == 5.0

    def test_delete_wallet_info(self, session):
        """Delete an entry"""
        wallet_info = WalletInfo(blockNumber=1, tokenValue=3.2)
        session.add(wallet_info)
        session.commit()

        session.delete(wallet_info)
        session.commit()

        deleted_wallet_info = session.query(WalletInfo).filter_by(blockNumber=1).first()
        assert deleted_wallet_info is None


class TestWalletInfoInterface:
    """Testing postgres interface for walletinfo table"""

    def test_latest_block_number(self, session):
        """Testing retrevial of wallet info via interface"""
        wallet_info_1 = WalletInfo(blockNumber=1, tokenValue=3.0)  # add your other columns here...
        postgres.add_wallet_infos([wallet_info_1], session)

        latest_block_number = postgres._get_latest_block_number_wallet_info(session)
        assert latest_block_number == 1

        wallet_info_2 = WalletInfo(blockNumber=2, tokenValue=3.2)  # add your other columns here...
        wallet_info_3 = WalletInfo(blockNumber=3, tokenValue=3.4)  # add your other columns here...
        postgres.add_wallet_infos([wallet_info_2, wallet_info_3], session)

        latest_block_number = postgres._get_latest_block_number_wallet_info(session)
        assert latest_block_number == 3

    def test_get_wallet_info(self, session):
        """Testing retrevial of walletinfo via interface"""
        wallet_info_1 = WalletInfo(blockNumber=0, tokenValue=3.1)  # add your other columns here...
        wallet_info_2 = WalletInfo(blockNumber=1, tokenValue=3.2)  # add your other columns here...
        wallet_info_3 = WalletInfo(blockNumber=2, tokenValue=3.3)  # add your other columns here...
        postgres.add_wallet_infos([wallet_info_1, wallet_info_2, wallet_info_3], session)

        wallet_info_df = postgres.get_all_wallet_info(session)
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.1, 3.2, 3.3], name="tokenValue"))

    def test_block_query_wallet_info(self, session):
        """Testing querying by block number of wallet info via interface"""
        wallet_info_1 = WalletInfo(blockNumber=0, tokenValue=3.1)  # add your other columns here...
        wallet_info_2 = WalletInfo(blockNumber=1, tokenValue=3.2)  # add your other columns here...
        wallet_info_3 = WalletInfo(blockNumber=2, tokenValue=3.3)  # add your other columns here...
        postgres.add_wallet_infos([wallet_info_1, wallet_info_2, wallet_info_3], session)

        wallet_info_df = postgres.get_all_wallet_info(session, start_block=1)
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.2, 3.3], name="tokenValue"))

        wallet_info_df = postgres.get_all_wallet_info(session, start_block=-1)
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.3], name="tokenValue"))

        wallet_info_df = postgres.get_all_wallet_info(session, end_block=1)
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.1], name="tokenValue"))

        wallet_info_df = postgres.get_all_wallet_info(session, end_block=-1)
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.1, 3.2], name="tokenValue"))

        wallet_info_df = postgres.get_all_wallet_info(session, start_block=1, end_block=-1)
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.2], name="tokenValue"))

    def test_current_wallet_info(self, session):
        """Testing helper function to get current wallet values"""
        wallet_info_1 = WalletInfo(
            blockNumber=0, walletAddress="addr", tokenType="BASE", tokenValue=3.1
        )  # add your other columns here...
        wallet_info_2 = WalletInfo(
            blockNumber=1, walletAddress="addr", tokenType="LP", tokenValue=5.1
        )  # add your other columns here...
        postgres.add_wallet_infos([wallet_info_1, wallet_info_2], session)

        wallet_info_df = postgres.get_current_wallet_info(session).reset_index()
        assert wallet_info_df["tokenType"].equals(pd.Series(["BASE", "LP"], name="tokenType"))
        assert wallet_info_df["tokenValue"].equals(pd.Series([3.1, 5.1], name="tokenValue"))

        # E.g., block 2, wallet base tokens gets updated to 6.1
        wallet_info_3 = WalletInfo(
            blockNumber=2, walletAddress="addr", tokenType="BASE", tokenValue=6.1
        )  # add your other columns here...
        postgres.add_wallet_infos([wallet_info_3], session)
        wallet_info_df = postgres.get_current_wallet_info(session).reset_index()
        assert wallet_info_df["tokenType"].equals(pd.Series(["BASE", "LP"], name="tokenType"))
        assert wallet_info_df["tokenValue"].equals(pd.Series([6.1, 5.1], name="tokenValue"))
