# DQN Trading Agent

## Prerequisites

```
pip install requirements.txt
```

## Script
Train model
```
python dqn.py
```
## Results
Developed a Deep Q-Learning trading agent that traded currency exchanges (eur-gbp, eur-usd, eur-gbp) under a simulated environment. The design of the model was experimented with different reward and exploration systems to assess how the DQN model performed under different circumstances. A [report] has been written that provides an introduction to the background knowledge required to understand this project along with the results of the system designs and the finalised models. 

[report]: https://github.com/j-truong/disso/blob/master/report/Dissertation.pdf

To differentiate from exisiting trading strategies, the DQN model was developed with two streams of inputs to not only determine a trading action, but to also quantify the amount of units for each trade. Layers from variations of deep learning were experimented due to their proven capabilities in prediciting stock market prices. DQN_ANN dictates a DQN function approximator with only hidden layers, whereas DQN_CNN and DQN_RNN correspond to a function approximator with convolutional layers and LSTM layers, respectively. 

The overall results are displayed below where popular trading strategies, Mean Reversion and Moving Average Trading Strategy, were developed to compare against these models. The DQN trading agent was unable to out-perform these popular trading strategies in terms the final portfolio value, frequency of trades, and the amount of time required to decide an action, but most were able to become profitable towards the end of the timestep. 

![image](https://github.com/j-truong/disso/blob/master/images/5_eurgbp.png)
![image](https://github.com/j-truong/disso/blob/master/images/5_eurusd.png)
![image](https://github.com/j-truong/disso/blob/master/images/5_eurchf.png)





