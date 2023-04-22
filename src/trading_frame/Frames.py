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

        self.date = date
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
        self.max_periods = max_periods
        self.raw_datas = []
        self.periods = []
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

        self._aggregate_to_period(raw_data)
