import abc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RawDataInterface(abc.ABC):
    @abc.abstractmethod
    def get_raw(self) -> list:
        """ Return a list of raw parameters """

class Tick(RawDataInterface):
    """
    A Tick is a move of two possible price : Bid and Ask prices.
    Bid and Ask are the best prices in the orderbook, they move when the
    orderbook is updated.

    Note: It's important to remember that some market don't provide real Volume,
          for example in the Forex market real volume are accessible only with
          specifics module or provider.
    """
    def __init__(self, date: str, bid_price: float, ask_price: float, bid_volume: float, ask_volume: float) -> None:
        """
        Initialization

        Parameters:
            date         (str): The date when tick appear
            bid_price  (float): Best price of the Bid side in the orderbook
            ask_price  (float): Best price of the Ask side in the orderbook
            bid_volume (float): Volume at the best price of the Bid side in the orderbook
            ask_volume (float): Volume at the best price of the Ask side in the orderbook
        """
        super(Tick, self).__init__()

        self.date       = date
        self.bid_price  = bid_price
        self.ask_price  = ask_price
        self.bid_volume = bid_volume
        self.ask_volume = ask_volume

        if self.bid_price > self.ask_price:
            raise Exception("Bid price should be lower or equal to the Ask price")

    def get_raw(self):
        return [self.date, self.bid_price, self.ask_price, self.bid_volume, self.ask_volume]

class Trade(RawDataInterface):
    """
    A Trade is an executed transaction between two actors : Buy and Seller.
    One of the two actors is the Maker, the one who propose volume at a price level
    and the other actor is the Taker, the one who agreed to pay at this price market level
    """
    def __init__(self, date: str, price: float, volume: float, side: bool) -> None:
        """
        Initialization

        Parameters:
            date     (str): The date the transaction was executed
            price  (float): The price level at which the trade was executed
            volume (float): The number of assets traded
            side    (bool): The side of execution.
                            False the Taker was Selling, True the Taker was Buying
        """
        super(Trade, self).__init__()

        self.date = date
        self.price = price
        self.volume = volume
        self.side = side

    def get_raw(self):
        return [self.date, self.price, self.volume, self.side]

class Candle(RawDataInterface):
    def __init__(self, date: str, _open: float, high: float, low: float, close: float, volume: float) -> None:
        """
        A Candle is an aggregation of executed trades between a period of time or count.
        """
        super(Candle, self).__init__()
        self.date = date
        self.open = _open
        self.high = high
        self.low  = low
        self.close = close
        self.volume = volume

    def get_raw(self):
        return [self.date, self.open, self.high, self.low, self.close, self.volume]

class Frame:
    """
    A Frame is a respresentation of aggregated raw trading datas.
    In it total basic shape it is just a clone of stacked raw datas, but inherited
    the Frame can be a lot of things.

    For example a TimeFrame of 5-minute periods, or a VolumeFrame of 1000-units periods
    """
    def __init__(self, max_periods: int = 250, date_format: str = '%Y-%m-%dT%H:%M:%S.%fZ') -> None:
        """
        Initialization method

        Parameters:
            max_periods  (int): Maximum length of periods list
        """
        self.date_format       = date_format
        self.max_periods       = max_periods
        self.periods           = []
        self.feeding_type      = None
        self.on_close_function = None

    def set_on_close_function(self, fn):
        """ Set the on close function """
        self.on_close_function = fn

    def on_close(self):
        """ Execute if the on close function is set """
        if self.on_close_function:
            self.on_close_function(self)

    def to_numpy(self) -> np.ndarray:
        """ Returning a periods numpy array"""
        return np.array(self.periods)

    def to_pandas(self) -> pd.DataFrame:
        """ Returning a Pandas DataFrame """
        columns = None

        if self.feeding_type == Trade:
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Close Date', 'Tick Volume', 'Volume', 'Buyers', 'Sellers']
        else:
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Close Date', 'Volume']

        df = pd.DataFrame(self.periods, columns=columns)
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace = True)

        return df

    def define_close_date(self, raw_data):
        """
        Defining the close date.
        This method is meant to be overrided

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.
        """
        return None

    def is_new_period(self, raw_data) -> bool:
        """
        Know if the raw data will start a new period or not.
        This method is meant to be overrided.

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.

        Returns:
            is_new_period (bool): Is this this a new period or not.
        """
        return True

    def create_new_period(self, raw_data) -> None:
        """
        Creating a new period

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.
        """
        close_date = self.define_close_date(raw_data)

        if self.feeding_type == Tick:
            # Tick Period = [open_date, open_price, high_price, low_price, close_price, close_date, tick_volume]
            self.periods.append([raw_data[0], raw_data[1], raw_data[1], raw_data[1], raw_data[1], close_date, 1])

        elif self.feeding_type == Trade:
            # Trade Period = [open_date, open_price, high_price, low_price, close_price, close_date, tick_volume, real_volume, buyers, sellers]
            buyers = 0
            sellers = 0

            if raw_data[3]:
                buyers += 1
            else:
                sellers += 1

            self.periods.append([raw_data[0], raw_data[1], raw_data[1], raw_data[1], raw_data[1], close_date, 1, raw_data[2], buyers, sellers])

        elif self.feeding_type == Candle:
            self.periods.append([raw_data[0], raw_data[1], raw_data[2], raw_data[3], raw_data[4], close_date, raw_data[5]])

    def update_period(self, raw_data) -> None:
        """
        Updating the last period in periods table

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.
        """
        if self.feeding_type == Tick or self.feeding_type == Trade:
            if raw_data[1] > self.periods[-1][2]:
                self.periods[-1][2] = raw_data[1]

            if raw_data[1] < self.periods[-1][3]:
                self.periods[-1][3] = raw_data[1]

            self.periods[-1][4] = raw_data[1]
            self.periods[-1][6] += 1

            if self.feeding_type == Trade:
                # Trade Period = [open_date, open_price, high_price, low_price, close_price, close_date, tick_volume, real_volume, buyers, sellers]
                self.periods[-1][7] += raw_data[2]

                if raw_data[3]:
                    self.periods[-1][8] += 1
                else:
                    self.periods[-1][9] += 1

        elif self.feeding_type == Candle:
            if raw_data[2] > self.periods[-1][2]:
                self.periods[-1][2] = raw_data[2]

            if raw_data[3] < self.periods[-1][3]:
                self.periods[-1][3] = raw_data[3]

            self.periods[-1][4] = raw_data[4]
            self.periods[-1][6] = raw_data[5]

    def aggregate_to_period(self, raw_data) -> None:
        """
        Aggregate new raw data to actual last period or creating a new period
        with raw data

        Trade and Tick periods are different because Trade raw data provide real volume
        and side of taker (buy side or sell side).

        Parameters:
            raw_data (list): The raw data (tick or trade)
        """
        if len(self.periods) < 1 or self.is_new_period(raw_data):
            if len(self.periods) > 1:
                """ Executing the on close function """
                self.on_close()

            self.create_new_period(raw_data)
        else:
            self.update_period(raw_data)

    def feed(self, raw_data: Tick | Trade | Candle) -> None:
        """
        Feeding the Frame with raw trading data.

        When the Frame is feeded the first time, it store the type of raw data (tick or trade),
        to remember it for the next feed. So if you feed your frame with different data,
        feed method will raise an exception.
        If length of raw datas is greater than the maximum raw data, it will pop the first index.

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.
        """
        if len(self.periods) > 0 and type(raw_data) != self.feeding_type:
            raise Exception(f"Raw Data feeded is not a {self.feeding_type} type")

        if len(self.periods) < 1:
            self.feeding_type = type(raw_data)

        self.aggregate_to_period(raw_data.get_raw())

        if self.max_periods < len(self.periods):
            self.periods.pop(0)

class TimeFrame(Frame):
    """
    TimeFrame base their data aggregation on Open Date or Close Date.
    """
    accepted_range = {
        'S': 'second',
        'T': 'minute',
        'H': 'hour',
        'D': 'day'
    }

    def __init__(self, periods_length: str = '5T', max_periods: int = 250, date_format: str = '%Y-%m-%dT%H:%M:%S.%fZ') -> None:
        """
        Initialization

        Parameters:
            periods_length (str): Maximum Length of every period.
            max_raw_data (int): Maximum length of raw_datas table
            max_periods  (int): Maximum length of periods table

        Returns:
            None
        """
        super(TimeFrame, self).__init__(max_periods, date_format)
        self.length = int(periods_length[0:-1])
        self.alias = periods_length[-1]

    def is_new_period(self, raw_data) -> bool:
        """
        Compare the raw data Date to the last period close date and return true
        if it's time to open a new period accordingly to the period length.

        Parameters:
            raw_data (list): The raw data (tick or trade)

        Returns:
            is_new_period (bool): Is this a new period or not
        """
        if len(self.periods) < 1:
            raise Exception("Periods table need at leat 1 period to compare to.")

        return datetime.strptime(raw_data[0], self.date_format) > datetime.strptime(self.periods[-1][5], self.date_format)

    def define_close_date(self, raw_data) -> str:
        """
        Defining the close date from the raw_data Date.

        Parameters:
            raw_data (list): The raw data (tick or trade)

        Returns:
            close_date (str): The close date for the period
        """
        open_date = datetime.strptime(raw_data[0], self.date_format)
        zeroing = getattr(open_date, self.accepted_range[self.alias]) % self.length

        if self.alias == 'S':
            open_date = open_date.replace(microsecond=0)
            close_date = (open_date - timedelta(seconds=zeroing)) + timedelta(seconds=self.length, microseconds=-1)

        if self.alias == 'T':
            open_date = open_date.replace(second=0, microsecond=0)
            close_date = (open_date - timedelta(minutes=zeroing)) + timedelta(minutes=self.length, microseconds=-1)

        if self.alias == 'H':
            open_date = open_date.replace(minute=0, second=0, microsecond=0)
            close_date = (open_date - timedelta(hours=zeroing)) + timedelta(hours=self.length, microseconds=-1)

        if self.alias == 'D':
            open_date = open_date.replace(hour=0, minute=0, second=0, microsecond=0)
            close_date = (open_date - timedelta(days=zeroing)) + timedelta(days=self.length, microseconds=-1)

        return close_date.strftime(self.date_format)

class CountFrame(Frame):
    """
    CountFrame base their data aggregation on number of a selected unit. It can be :
    - Tick Volume : the number of trades executed (tick_volume)
    - Real Volume : the volume exchanged (real_volume)

    """
    def __init__(self, length: int, counted: str = "tick_volume", max_periods: int = 250) -> None:
        """
        Initialization

        Parameters:
            length       (int): Length of maximum unit to count
            counted      (str): The value to count
            max_raw_data (int): Maximum length of raw_datas table
            max_periods  (int): Maximum length of periods table
        """
        super(CountFrame, self).__init__(max_periods)
        self.length = length
        self.counted = counted

    def is_new_period(self, raw_data) -> bool:
        """
        Know if the raw data will start a new period or not.
        This method is meant to be overrided.

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.

        Returns:
            is_new_period (bool): Is this this a new period or not.
        """
        if self.counted == "tick_volume":
            return self.periods[-1][6] + 1 > self.length

        if self.counted == "real_volume":
            return self.periods[-1][7] + raw_data[2] > self.length

    def define_close_date(self, raw_data) -> None:
        """
        Defining the close date from the raw_data Date.

        Parameters:
            raw_data (list): The raw data (tick or trade)
        """
        if len(self.periods) > 0:
            self.periods[-1][5] = raw_data[0]

        return None
