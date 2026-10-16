import unittest
from fabsense import capability, gauge_rr, rul_baseline, simulate_train, utility_anomalies


class FabTests(unittest.TestCase):
    def test_rul(self):
        self.assertEqual(rul_baseline([1, 2, 3]), [2, 1, 0])

    def test_anomaly(self):
        self.assertIn(10, utility_anomalies([10, 11]*5 + [100]))

    def test_sim(self):
        self.assertEqual(simulate_train(4), simulate_train(4))
        self.assertGreater(simulate_train(4)["throughput_lots_per_hour"], 0)

    def test_measurement(self):
        self.assertGreater(capability([9, 10, 11], 5, 15)["cpk"], 0)
        self.assertGreater(gauge_rr()["repeatability_sd_mean"], 0)


if __name__ == "__main__":
    unittest.main()
