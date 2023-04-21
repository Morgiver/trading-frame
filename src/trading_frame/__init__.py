# Custom Python
from src.trading_frame.Factory import ViewFactory
from src.trading_frame.Views import View
from src.trading_frame.Frames import Tick, Trade

class ViewManager:
    def __init__(self):
        self.views = {}

    def add(self, name: str, builded_view: View):
        if not name in self.views:
            self.views[name] = builded_view
            return self

        raise Exception(f"A view with name : [{name}] already exist")

    def feed(self, raw_data: list, _type: str = "tick"):
        pass
