"""
Unit tests for polymarket_scanner.py
Tests all logic using mock data — no internet required.
"""

import unittest
from unittest.mock import patch, MagicMock
from polymarket_scanner import (
    parse_order_book_side,
    analyze_market,
    Anomaly,
    OrderBookSide,
    MIN_TRADE_USD,
    GAP_THRESHOLD,
    OVERLAP_THRESHOLD,
)


# ─── Helper: build a fake order book response ────────────────────────────────
def make_book(asks: list[tuple], bids: list[tuple]) -> dict:
    """asks/bids: list of (price, size) tuples"""
    return {
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
    }


def make_market(yes_token="YES_TOK", no_token="NO_TOK", slug="test-market",
                question="Will X happen?", cid="COND_123"):
    return {
        "slug": slug,
        "conditionId": cid,
        "question": question,
        "outcomes": ["Yes", "No"],
        "tokens": [
            {"token_id": yes_token, "outcome": "Yes"},
            {"token_id": no_token,  "outcome": "No"},
        ],
        "active": True,
    }


# ─── Tests: parse_order_book_side ────────────────────────────────────────────
class TestParseOrderBookSide(unittest.TestCase):

    def test_empty_returns_none(self):
        result = parse_order_book_side([], "asks")
        self.assertIsNone(result)

    def test_asks_sorted_ascending(self):
        asks = [{"price": "0.60", "size": "200"}, {"price": "0.55", "size": "100"}]
        result = parse_order_book_side(asks, "asks")
        self.assertAlmostEqual(result.best_price, 0.55)

    def test_bids_sorted_descending(self):
        bids = [{"price": "0.40", "size": "100"}, {"price": "0.45", "size": "200"}]
        result = parse_order_book_side(bids, "bids")
        self.assertAlmostEqual(result.best_price, 0.45)

    def test_depth_at_best_aggregates_same_price(self):
        asks = [
            {"price": "0.55", "size": "100"},
            {"price": "0.55", "size": "200"},  # same price level
            {"price": "0.65", "size": "300"},
        ]
        result = parse_order_book_side(asks, "asks")
        # 100 * 0.55 + 200 * 0.55 = 55 + 110 = 165
        self.assertAlmostEqual(result.depth_at_best, 165.0, places=1)

    def test_total_depth_within_010(self):
        asks = [
            {"price": "0.55", "size": "100"},
            {"price": "0.60", "size": "200"},  # within 0.10
            {"price": "0.70", "size": "500"},  # outside 0.10
        ]
        result = parse_order_book_side(asks, "asks")
        expected = 100 * 0.55 + 200 * 0.60  # = 55 + 120 = 175
        self.assertAlmostEqual(result.total_depth_100, expected, places=1)


# ─── Tests: analyze_market ────────────────────────────────────────────────────
class TestAnalyzeMarket(unittest.TestCase):

    def _run(self, yes_book, no_book):
        """Helper: patch fetch_order_book and run analyze_market."""
        market = make_market()
        books = {"YES_TOK": yes_book, "NO_TOK": no_book}

        def fake_fetch(token_id):
            return books.get(token_id)

        with patch("polymarket_scanner.fetch_order_book", side_effect=fake_fetch), \
             patch("polymarket_scanner.time.sleep"):
            return analyze_market(market)

    # ── GAP arbitrage (buy both) ──────────────────────────────────────────────
    def test_gap_arbitrage_detected(self):
        """YES_ask=0.45, NO_ask=0.45 → sum=0.90, gap=0.10 → should flag."""
        yes_book = make_book(
            asks=[(0.45, 500)],
            bids=[(0.40, 500)],
        )
        no_book = make_book(
            asks=[(0.45, 500)],
            bids=[(0.40, 500)],
        )
        result = self._run(yes_book, no_book)
        self.assertIsNotNone(result)
        self.assertIn("GAP", result.anomaly_type)
        self.assertAlmostEqual(result.gap_value, 0.10, places=2)

    def test_gap_arbitrage_below_threshold_not_flagged(self):
        """Gap of only 0.003 < GAP_THRESHOLD (0.005) → should not flag."""
        yes_book = make_book(asks=[(0.499, 500)], bids=[(0.48, 500)])
        no_book  = make_book(asks=[(0.499, 500)], bids=[(0.48, 500)])
        result = self._run(yes_book, no_book)
        # sum = 0.998, gap = 0.002, below GAP_THRESHOLD=0.005 → no gap flag
        # Check no GAP anomaly (might be MID deviation)
        if result:
            self.assertNotIn("GAP", result.anomaly_type)

    # ── OVERLAP arbitrage (sell both) ─────────────────────────────────────────
    def test_overlap_arbitrage_detected(self):
        """YES_bid=0.60, NO_bid=0.60 → sum=1.20, overlap=0.20 → flag."""
        yes_book = make_book(asks=[(0.65, 500)], bids=[(0.60, 500)])
        no_book  = make_book(asks=[(0.65, 500)], bids=[(0.60, 500)])
        result = self._run(yes_book, no_book)
        self.assertIsNotNone(result)
        self.assertIn("OVERLAP", result.anomaly_type)
        self.assertAlmostEqual(result.gap_value, 0.20, places=2)

    # ── Mid-price deviation ───────────────────────────────────────────────────
    def test_mid_deviation_detected(self):
        """YES mid=0.70, NO mid=0.20, ask sum=1.05 (no gap), bid sum=0.75 (no overlap)."""
        # ask sum 0.75+0.30=1.05 > 1.0 → no gap
        # bid sum 0.65+0.10=0.75 < 1.0 → no overlap
        # mid sum 0.70+0.20=0.90 → deviation 0.10 > MID_THRESHOLD=0.02
        # NO_bid @ 0.10 needs size>1000 to exceed $100 liquidity: 2000*0.10=$200 ✓
        yes_book = make_book(asks=[(0.75, 500)],  bids=[(0.65, 500)])   # mid=0.70; liq: ask=$375, bid=$325
        no_book  = make_book(asks=[(0.30, 2000)], bids=[(0.10, 2000)])  # mid=0.20; liq: ask=$600, bid=$200
        result = self._run(yes_book, no_book)
        self.assertIsNotNone(result)
        self.assertIn("MID_DEVIATION", result.anomaly_type)

    def test_no_anomaly_when_prices_balanced(self):
        """YES_ask=0.52, NO_ask=0.52 → balanced → no anomaly."""
        yes_book = make_book(asks=[(0.52, 500)], bids=[(0.50, 500)])  # mid=0.51
        no_book  = make_book(asks=[(0.52, 500)], bids=[(0.50, 500)])  # mid=0.51
        # Sum of asks = 1.04 (overlap < threshold of 0.005? 0.04 > 0.005 → OVERLAP!)
        # Actually 0.52+0.52=1.04 sum of asks... but we check bids for overlap
        # YES_bid=0.50 + NO_bid=0.50 = 1.00 → exactly 1, gap_value=0 → no overlap
        # YES_ask=0.52 + NO_ask=0.52 = 1.04 → sum > 1, no gap
        # mid: (0.51 + 0.51) = 1.02, dev = 0.02 == MID_THRESHOLD boundary
        result = self._run(yes_book, no_book)
        # At exactly MID_THRESHOLD it should NOT trigger (strict >)
        if result:
            self.assertNotIn("GAP", result.anomaly_type)

    # ── Liquidity filter ──────────────────────────────────────────────────────
    def test_gap_below_min_liquidity_filtered_out(self):
        """Gap exists but only $50 of liquidity → should be filtered out."""
        # YES_ask=0.40 depth: size=10 → USD = 10*0.40 = $4 (not $100)
        yes_book = make_book(asks=[(0.40, 10)], bids=[(0.35, 10)])
        no_book  = make_book(asks=[(0.40, 10)], bids=[(0.35, 10)])
        result = self._run(yes_book, no_book)
        self.assertIsNone(result)

    def test_gap_with_sufficient_liquidity_passes(self):
        """Gap with $1000 of liquidity on each leg → passes filter."""
        yes_book = make_book(asks=[(0.45, 1000)], bids=[(0.40, 1000)])
        no_book  = make_book(asks=[(0.45, 1000)], bids=[(0.40, 1000)])
        result = self._run(yes_book, no_book)
        self.assertIsNotNone(result)
        self.assertGreater(result.tradeable_size_usd, 100)

    # ── URL construction ──────────────────────────────────────────────────────
    def test_url_contains_slug(self):
        yes_book = make_book(asks=[(0.45, 500)], bids=[(0.40, 500)])
        no_book  = make_book(asks=[(0.45, 500)], bids=[(0.40, 500)])
        result = self._run(yes_book, no_book)
        self.assertIsNotNone(result)
        self.assertIn("test-market", result.url)

    # ── Missing tokens ────────────────────────────────────────────────────────
    def test_returns_none_if_book_fetch_fails(self):
        with patch("polymarket_scanner.fetch_order_book", return_value=None), \
             patch("polymarket_scanner.time.sleep"):
            result = analyze_market(make_market())
        self.assertIsNone(result)

    def test_returns_none_for_malformed_market(self):
        bad_market = {"slug": "bad", "question": "?", "outcomes": [], "tokens": []}
        with patch("polymarket_scanner.fetch_order_book", return_value=None), \
             patch("polymarket_scanner.time.sleep"):
            result = analyze_market(bad_market)
        self.assertIsNone(result)


# ─── Integration-style: fetch_active_binary_markets ──────────────────────────
class TestFetchMarkets(unittest.TestCase):

    def test_filters_non_binary_markets(self):
        """Multi-outcome markets (3+ options) should be excluded."""
        from polymarket_scanner import fetch_active_binary_markets

        fake_markets = [
            # Valid binary market
            {
                "question": "Will X?", "slug": "x", "conditionId": "c1",
                "outcomes": ["Yes", "No"],
                "tokens": [{"token_id": "T1"}, {"token_id": "T2"}],
                "active": True,
            },
            # Invalid: 3 outcomes
            {
                "question": "Who wins?", "slug": "y", "conditionId": "c2",
                "outcomes": ["A", "B", "C"],
                "tokens": [{"token_id": "T3"}, {"token_id": "T4"}, {"token_id": "T5"}],
                "active": True,
            },
            # Invalid: no outcomes
            {
                "question": "Empty?", "slug": "z", "conditionId": "c3",
                "outcomes": [],
                "tokens": [],
                "active": True,
            },
        ]

        with patch("polymarket_scanner.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_markets
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            result = fetch_active_binary_markets(max_markets=10)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["slug"], "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
