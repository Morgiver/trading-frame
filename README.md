# Trading Frame

This package will provide an API to manage Frame of trading data.

Frames are data organised in periods table. There's multiple frame's type like Time Frame, Tick Frame, Volume Frame, Range Frame, Reversal Frame (and surely other).
The objective of this package is to manage easely every type of frame, feeded by every type of trading data. In the end, the user should be capable to request a global view of organised trading data.

## Actual State : Iteration 0 (see [historic for past iterations](https://github.com/Morgiver/trading-frame/blob/main/iterations.md))

1. Make unit test, to ensure the stability of the project, for all code created.
2. ~~Define and build a Frame class to inherit and build other Frame type with. Frame are defining how the raw data will be feeded and aggregated, for example we want a normal Candlestick chart with data organised in 5 minutes periods, we will use a Candlestick view exploiting a TimeFrame. So Frame will stack raw data and send them to a view to be organised.~~
3. Define and build a View class to inherit and build other View type with. Different type of view are : Simple Bar, Candlestick, Renko, Heiken Ashi, Kagi, Point and Figures, and I'm sure many other. For this iteration 0, one or two View should be cool. View should simply exploit the data organised in it's Frame.
4. Define and build a ViewFactory able to create multiple views with their frame.
5. Define and build a ViewManager able to stack multiple named views and manage them. Managing views is create, update, delete and feed.
6. ~~Define and build TimeFrame to be able to handle time based periods~~
7. ~~Define and build CountFrame to be able to handle count based ticks or trades periods~~

### Idea for Iteration 1
1. Allow Candle raw data for Frame.

### Important Note :
1. Will be logic to establish a standard date format system in futures iteration
