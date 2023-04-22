import unittest

from src.trading_frame.Frames import *

class TestTick(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(Exception):
            Tick('01/01/2000, 00:00:00', 1.1, 1.0, 1.0, 2.0)

    def test_get_raw(self):
        t = Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0)
        self.assertEqual(type(t.get_raw()), list)

class TestTrade(unittest.TestCase):
    def test_get_raw(self):
        t = Trade('01/01/2000, 00:00:00', 1.0, 1.0, True)
        self.assertEqual(type(t.get_raw()), list)

class TestFrame(unittest.TestCase):
    def test_feed(self):
        f = Frame(max_raw_data = 1)
        f.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        self.assertEqual(len(f.raw_datas), 1)
        f.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        self.assertEqual(len(f.raw_datas), 1)

    def test_feed_raising(self):
        with self.assertRaises(Exception):
            f = Frame(max_raw_data = 1)
            f.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
            f.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
