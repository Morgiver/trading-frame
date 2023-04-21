# Custom Python
from src.trading_frame.Frames import Frame

class View:
    def __init__(self) -> None:
        self.frame = None

    def set_frame(self, frame: Frame):
        pass

class CandlestickView(View) -> None:
    def __init__(self):
        super(CandlestickView, self).__init__()
