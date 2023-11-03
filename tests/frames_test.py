import sys
import unittest
from types import NoneType

sys.path.append('./src')

# Custom Python
from trading_frame.Frames import *

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
        f = Frame(max_periods = 1)
        f.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        self.assertEqual(len(f.periods), 1)
        f.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        self.assertEqual(len(f.periods), 1)

    def test_feed_raising(self):
        with self.assertRaises(Exception):
            f = Frame(max_periods = 1)
            f.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
            f.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))

class TestTimeFrame(unittest.TestCase):
    def test_define_close_date_seconds(self):
        tf = TimeFrame(periods_length='5S')
        date_to_compare = tf.define_close_date(Trade('01/01/2000, 00:02:12', 1.0, 1.0, True).get_raw())
        compare_time = datetime.strptime('01/01/2000, 00:02:15', DATE_STR_FORMAT) - timedelta(microseconds=1)
        self.assertEqual(date_to_compare, compare_time.strftime(DATE_STR_FORMAT))

    def test_define_close_date_minutes(self):
        tf = TimeFrame(periods_length='5T')
        date_to_compare = tf.define_close_date(Trade('01/01/2000, 00:02:12', 1.0, 1.0, True).get_raw())
        compare_time = datetime.strptime('01/01/2000, 00:05:00', DATE_STR_FORMAT) - timedelta(microseconds=1)
        self.assertEqual(date_to_compare, compare_time.strftime(DATE_STR_FORMAT))

    def test_define_close_date_hours(self):
        tf = TimeFrame(periods_length='1H')
        date_to_compare = tf.define_close_date(Trade('01/01/2000, 00:02:12', 1.0, 1.0, True).get_raw())
        compare_time = datetime.strptime('01/01/2000, 01:00:00', DATE_STR_FORMAT) - timedelta(microseconds=1)
        self.assertEqual(date_to_compare, compare_time.strftime(DATE_STR_FORMAT))

    def test_define_close_date_days(self):
        tf = TimeFrame(periods_length='1D')
        date_to_compare = tf.define_close_date(Trade('01/01/2000, 00:02:12', 1.0, 1.0, True).get_raw())
        compare_time = datetime.strptime('01/02/2000, 00:00:00', DATE_STR_FORMAT) - timedelta(microseconds=1)
        self.assertEqual(date_to_compare, compare_time.strftime(DATE_STR_FORMAT))

    def test_is_new_period(self):
        tf = TimeFrame()
        tf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        self.assertEqual(tf.is_new_period(Trade('01/01/2000, 00:05:00', 1.0, 1.0, True).get_raw()), True)

    def test_is_new_period_raising(self):
        tf = TimeFrame()
        with self.assertRaises(Exception):
            tf.is_new_period(Trade('01/01/2000, 00:05:00', 1.0, 1.0, True).get_raw())

    def test_aggregate_to_period_trade(self):
        tf = TimeFrame()
        tf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.0, 1.0, 1.0, '01/01/2000, 00:04:59', 1, 1.0, 1, 0]])
        tf.feed(Trade('01/01/2000, 00:02:25', 1.5, 1.0, True))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 1.0, 1.5, '01/01/2000, 00:04:59', 2, 2.0, 2, 0]])
        tf.feed(Trade('01/01/2000, 00:03:25', 0.5, 1.0, False))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 0.5, 0.5, '01/01/2000, 00:04:59', 3, 3.0, 2, 1]])
        tf.feed(Trade('01/01/2000, 00:04:25', 1.25, 1.0, True))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 0.5, 1.25, '01/01/2000, 00:04:59', 4, 4.0, 3, 1]])
        tf.feed(Trade('01/01/2000, 00:05:00', 1.35, 1.0, True))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 0.5, 1.25, '01/01/2000, 00:04:59', 4, 4.0, 3, 1], ['01/01/2000, 00:05:00', 1.35, 1.35, 1.35, 1.35, '01/01/2000, 00:09:59', 1, 1.0, 1, 0]])

    def test_aggregate_to_period_tick(self):
        tf = TimeFrame()
        tf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.0, 1.0, 1.0, '01/01/2000, 00:04:59', 1]])
        tf.feed(Tick('01/01/2000, 00:02:25', 1.5, 1.6, 1.0, 2.0))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 1.0, 1.5, '01/01/2000, 00:04:59', 2]])
        tf.feed(Tick('01/01/2000, 00:03:25', 0.5, 0.6, 1.0, 2.0))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 0.5, 0.5, '01/01/2000, 00:04:59', 3]])
        tf.feed(Tick('01/01/2000, 00:04:25', 1.25, 1.35, 1.0, 2.0))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 0.5, 1.25, '01/01/2000, 00:04:59', 4]])
        tf.feed(Tick('01/01/2000, 00:05:00', 1.35, 1.45, 1.0, 2.0))
        self.assertEqual(tf.periods, [['01/01/2000, 00:00:00', 1.0, 1.5, 0.5, 1.25, '01/01/2000, 00:04:59', 4], ['01/01/2000, 00:05:00', 1.35, 1.35, 1.35, 1.35, '01/01/2000, 00:09:59', 1]])

class TestCountFrame(unittest.TestCase):
    def test_is_new_period_tick_volume(self):
        cf = CountFrame(5, "tick_volume")
        cf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        cf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        cf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        cf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        cf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        cf.feed(Tick('01/01/2000, 00:00:00', 1.0, 1.1, 1.0, 2.0))
        self.assertEqual(cf.periods[0][6], 5)

    def test_is_new_period_real_volume(self):
        cf = CountFrame(5, "real_volume")
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        self.assertEqual(cf.periods[0][7], 5.0)

    def test_define_close_date(self):
        cf = CountFrame(1, "real_volume")
        result = cf.define_close_date(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True).get_raw())
        self.assertEqual(type(result), NoneType)
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        cf.feed(Trade('01/01/2000, 00:00:00', 1.0, 1.0, True))
        self.assertEqual(cf.periods[0][5], '01/01/2000, 00:00:00')
