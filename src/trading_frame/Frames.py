from datetime import datetime, timedelta

DATE_STR_FORMAT = '%m/%d/%Y, %H:%M:%S'

class RawDataInterface:
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

        Returns:
            None
        """
        super(Tick, self).__init__()

        self.date       = date
        self.bid_price  = bid_price
        self.ask_price  = ask_price
        self.bid_volume = bid_volume
        self.ask_volume = ask_volume

        if self.bid_price > self.ask_price:
            raise Execption("Bid price should be lower or equal to the Ask price")

    def get_raw(self):
        return [self.date, self.bid_price, self.ask_price, self.bid_volume, self.ask_volume]

class Trade(RawDataInterface):
    """
    A Trade is an executed transaction between two actors : Buy and Seller.
    One of the two actors is the Maker, the one who propose volume at a price level
    and the other actor is the Taker, the one who agreed to pay at this price market level
    """
    def __init__(self, date: str, price: float, volume: float, taker: bool) -> None:
        """
        Initialization

        Parameters:
            date     (str): The date the transaction was executed
            price  (float): The price level at which the trade was executed
            volume (float): The number of assets traded
            side    (bool): The side of execution.
                            False the Taker was Selling, True the Taker was Buying
        Returns:
            None
        """
        super(Trade, self).__init__()

        self.date = date
        self.price = price
        self.volume = volume
        self.side = taker

    def get_raw(self):
        return [self.date, self.price, self.volume, self.side]

class Frame:
    """
    A Frame is a respresentation of aggregated raw trading datas.
    In it total basic shape it is just a clone of stacked raw datas, but inherited
    the Frame can be a lot of things.

    For example a TimeFrame of 5-minute periods, or a VolumeFrame of 1000-units periods
    """
    def __init__(self, max_raw_data: int = 1000, max_periods: int = 250) -> None:
        """
        Initialization method

        Parameters:
            max_raw_data (int): Maximum length of raw_datas list
            max_periods  (int): Maximum length of periods list

        Returns:
            None
        """
        self.max_raw_data = max_raw_data
        self.max_periods  = max_periods
        self.raw_datas    = []
        self.periods      = []
        self.feeding_type = None

    def _aggregate_to_period(self, raw_data: Tick | Trade) -> None:
        """
        Aggregation method
        This is the function to override to define an appropriate aggregation method

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.

        Returns:
            None
        """
        pass

    def feed(self, raw_data: Tick | Trade) -> None:
        """
        Feeding the Frame with raw trading data.

        When the Frame is feeded the first time, it store the type of raw data (tick or trade),
        to remember it for the next feed. So if you feed your frame with different data,
        feed method will raise an exception.
        If length of raw datas is greater than the maximum raw data, it will pop the first index.

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.
        Returns:
            None
        """
        if len(self.raw_datas) > 0 and type(raw_data) != self.feeding_type:
            raise Exception(f"Raw Data feeded is not a {self.feeding_type} type")

        if len(self.raw_datas) < 1:
            self.feeding_type = type(raw_data)

        self.raw_datas.append(raw_data.get_raw())

        if len(self.raw_datas) > self.max_raw_data:
            self.raw_datas.pop(0)

        self._aggregate_to_period(raw_data.get_raw())

class TimeFrame(Frame):
    """
    TimeFrame base their data aggregation on Open Date or Close Date.
    """
    def __init__(self, periods_length: str = '5T', max_raw_data: int = 1000, max_periods: int = 250) -> None:
        """
        Initialization

        Parameters:
            periods_length (int): Maximum Length of every period (in second).

        Returns:
            None
        """
        super(TimeFrame, self).__init__(max_raw_data, max_periods)
        self.length = int(periods_length[0])
        self.alias = periods_length[1]

        print(self.alias)

        self.accepted_range = {
            'S': 'second',
            'T': 'minute',
            'H': 'hour',
            'D': 'day'
        }

    def is_new_period(self, raw_data):
        """
        Compare the raw data Date to the last period close date and return true
        if it's time to open a new period accordingly to the period length.

        Parameters:
            raw_data (list): The raw data (tick or trade)

        Returns:
            is_new_period (bool): Is this a new periods or not
        """
        if len(self.periods) < 1:
            raise Exception("Periods table need at leat 1 period to compare to.")

        return datetime.strptime(raw_data[0], DATE_STR_FORMAT) > datetime.strptime(self.periods[-1][5], DATE_STR_FORMAT)

    def define_close_date(self, raw_data):
        """
        Defining the close date from the raw_data Date.

        Parameters:
            raw_data (list): The raw data (tick or trade)

        Returns:
            close_date (Datetime): The close date for the period
        """
        open_date = datetime.strptime(raw_data[0], DATE_STR_FORMAT)
        zeroing = getattr(open_date, self.accepted_range[self.alias]) % self.length

        if self.alias == 'S':
            open_date = open_date.replace(microsecond=0)
            return (open_date - timedelta(seconds=zeroing)) + timedelta(seconds=self.length, microseconds=-1)

        if self.alias == 'T':
            open_date = open_date.replace(second=0, microsecond=0)
            return (open_date - timedelta(minutes=zeroing)) + timedelta(minutes=self.length, microseconds=-1)

        if self.alias == 'H':
            open_date = open_date.replace(minute=0, second=0, microsecond=0)
            return (open_date - timedelta(hours=zeroing)) + timedelta(hours=self.length, microseconds=-1)

        if self.alias == 'D':
            open_date = open_date.replace(hour=0, minute=0, second=0, microsecond=0)
            return (open_date - timedelta(days=zeroing)) + timedelta(days=self.length, microseconds=-1)

    def _aggregate_to_period(self, raw_data):
        """
        Aggregate new raw data to actual last period or creating a new period
        with raw data

        Parameters:
            raw_data (list): The raw data (tick or trade)

        Returns:
            None
        """
        if len(self.periods) < 1 or self.is_new_period(raw_data):
            close_date = self.define_close_date(raw_data).strftime(DATE_STR_FORMAT)

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
        else:
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
