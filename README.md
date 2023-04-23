# Trading Frame

This package will provide an API to manage Frame of trading data.

Frames are data organised in periods table. There's multiple frame's type like Time Frame, Tick Frame, Volume Frame, Range Frame, Reversal Frame (and surely other).
The objective of this package is to manage easely every type of frame, feeded by every type of trading data. In the end, the user should be capable to request a global view of organised trading data.

## Actual State : Iteration 1 [Working on definition] (see [historic for past iterations](https://github.com/Morgiver/trading-frame/blob/main/iterations.md))

1. Allow Candle raw data for Frame.
2. Define and build a View class to inherit and build other View type with. Different type of view are : Simple Bar, Candlestick, Renko, Heiken Ashi, Kagi, Point and Figures, and I'm sure many other. For this iteration 0, one or two View should be cool. View should simply exploit the data organised in it's Frame.
3. Define and build a ViewFactory able to create multiple views with their frame.
4. Define and build a ViewManager able to stack multiple named views and manage them. Managing views is create, update, delete and feed.
