## Iteration 0
1. ~~Make unit test, to ensure the stability of the project, for all code created.~~
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

## Iteration 1
There was a large time gap in this iteration and many change have come. I wasn't disciplined enough and i work directly on different ideas. So i will resume what compose the project at this time.

1. First we have RawDataInterface type, this ensure to build different type of RawData that can be identified.
   For now we have : Tick, Trade and Candle that extend RawDataInterface
2. Of course we have Tick, Trade and Candle classes, they build the structure for basic datas that feed the frame.
3. Frame class compose the base class to create different type of Frame. A Frame is container and manager for a range of periods. Its role is to feed and build the periods chronologically.
   Every Frame has a little event manager that emit 'close' and 'update' events. It allow developpers to execute function at the right time
4. TimeFrame extend Frame base class, it build a time based frame. This is the most common Frame used by the majority of traders.
5. CountFrame extend Frame base class, it build a counted value based frame. The counted value can be the number of trade executed, a volume limit value, a number of tick, etc.

### What's coming for the next iteration ?
My idea is to focus on the possibility to handle some tool that can manage additionnal value processing on different levels.
1. First i want to allow developpers to add columns on the main periods data table. E.g if the developper want to add the process of RSI indicator, or the different element of MACD, etc. any indicator that can fit in the main data table.
2. The second level is to add the possibility to create parallel data table. E.g if the developper want to process small data table that contain highs and lows. Or the possibility to process Volume Profile, etc.
