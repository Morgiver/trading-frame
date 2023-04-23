## Iteration 0
1. Make unit test, to ensure the stability of the project, for all code created.
2. ~~Define and build a Frame class to inherit and build other Frame type with. Frame are defining how the raw data will be feeded and aggregated, for example we want a normal Candlestick chart with data organised in 5 minutes periods, we will use a Candlestick view exploiting a TimeFrame. So Frame will stack raw data and send them to a view to be organised.~~
3. Define and build a View class to inherit and build other View type with. Different type of view are : Simple Bar, Candlestick, Renko, Heiken Ashi, Kagi, Point and Figures, and I'm sure many other. For this iteration 0, one or two View should be cool. View should simply exploit the data organised in it's Frame.
4. Define and build a ViewFactory able to create multiple views with their frame.
5. Define and build a ViewManager able to stack multiple named views and manage them. Managing views is create, update, delete and feed.
6. ~~Define and build TimeFrame to be able to handle time based periods~~
7. ~~Define and build CountFrame to be able to handle count based ticks or trades periods~~

### Points for next Iteration
1. Allow Candle raw data for Frame.
2. Define and build a View class to inherit and build other View type with. Different type of view are : Simple Bar, Candlestick, Renko, Heiken Ashi, Kagi, Point and Figures, and I'm sure many other. For this iteration 0, one or two View should be cool. View should simply exploit the data organised in it's Frame.
3. Define and build a ViewFactory able to create multiple views with their frame.
4. Define and build a ViewManager able to stack multiple named views and manage them. Managing views is create, update, delete and feed.

### Important Note :
1. Will be logic to establish a standard date format system in futures iteration
2. Point 3, 4, 5 are delayed to Iteration 1 since the actual iteration is working and has getting 2-3 days of work.
