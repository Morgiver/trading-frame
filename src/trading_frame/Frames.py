class Tick:
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
        self.date = date
        self.bid_price  = bid_price
        self.ask_price  = ask_price
        self.bid_volume = bid_volume
        self.ask_volume = ask_volume

class Trade:
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
        self.date = date
        self.price = price
        self.volume = Volume
        self.side = taker

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
        Feeding the Frame with raw data

        Parameters:
            raw_data (Tick | Trade): The raw data to work with.
                                     It can be a Tick or a Trade.
        Returns:
            None
        """
        if type(raw_data) == Tick:
            self.raw_datas.append([raw_data.date, raw_data.bid_price, raw_data.ask_price, raw_data.volume])
        if type(raw_data) == Trade:
            self.raw_datas.append([raw_data.date, raw_data.price, raw_data.volume, raw_data.side])

        if len(self.raw_datas) <= self.max_raw_data:
            self.raw_datas.pop(0)

        self._aggregate_to_period(raw_data)
