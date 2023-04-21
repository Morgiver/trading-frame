# Custom Python
from src.trading_frame.Views import *

class ViewFactory:
    @staticmethod
    def set_view(*args, **kwargs):
        pass

    @staticmethod
    def set_frame(*args, **kwargs):
        pass

    @staticmethod
    def build(*args, **kwargs):
        view = ViewFactory.set_view(**kwargs['view'])
        frame = ViewFactory.set_frame(**kwargs['frame'])
        view.set_frame(frame)
