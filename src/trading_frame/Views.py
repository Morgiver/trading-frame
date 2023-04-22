# Custom Python
from src.trading_frame.Frames import Frame

class View:
    def __init__(self) -> None:
        self.frame = None

    def set_frame(self, frame: Frame):
        pass

class CandlestickView(View):
    def __init__(self) -> None:
        super(CandlestickView, self).__init__()
