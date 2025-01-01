import unittest

from Market import Market


class TestMarket(unittest.TestCase):
    def setUp(self):
        self.market = Market()
        self.market.set_parameters(0.1, -0.5, 0.5, 0.005, 0.2, 0.1, 0.06, 10, 0.01)
        self.market.compile(100, 10, 5)  # Example parameters

    def test_initialization(self):
        self.assertIsInstance(self.market, Market)
        self.assertEqual(len(self.market.Companies), 0)

    def test_set_parameters(self):
        self.assertEqual(self.market.Selection_Pressure, 0.1)
        self.assertEqual(self.market.MinQuality, -0.5)

    def test_setup_companies(self):
        self.market.setup_companies()
        self.assertEqual(len(self.market.Companies), 5)

    def test_new_entry(self):
        self.market.setup_companies()
        initial_count = len(self.market.Companies)
        self.market.new_entry()
        self.assertTrue(len(self.market.Companies) >= initial_count)

    def test_dropout(self):
        self.market.setup_companies()
        # Artificially reduce a company's market share to force dropout
        self.market.Companies[0].MarketShare = 0
        self.market.dropout()
        self.assertTrue(len(self.market.Companies) < 5)

    def test_market_dynamics(self):
        self.market.setup_companies()
        initial_total_sales = self.market.Total_Sales
        self.market.step()
        # Assuming some market dynamics have occurred
        self.assertNotEqual(self.market.Total_Sales, initial_total_sales)


if __name__ == '__main__':
    unittest.main()
