# Custom Python
from src.trading_frame.Frames import Frame

class View:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame

    def set_frame(self, frame: Frame):
        pass
