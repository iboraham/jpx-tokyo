# JPX - Tokyo stock

---

## Steps

- [ ] Read all the discussions
- [ ] Read all the codes
- [ ] Define Problem
- [ ] Literature Review on the problem e.g paperswithcode.com
- [ ] Baseline Solution
- [ ] Feature Engineering e.g. work more on the data
- [ ] More complicated model or fine tune model

## Discussion notes

**- How to make submission?**

Here is an example code,

```python
import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test files
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    sample_prediction_df['Rank'] = np.arange(len(sample_prediction))  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
```

**- Previous Examples like this**

> Jquants -> https://www.jpx.co.jp/english/corporate/news/news-releases/0010/20210813-01.html

> Jane -> https://www.kaggle.com/competitions/jane-street-market-prediction

## Peer Code notes

## Plan

- [ ] Read data with fe.get_train_data()
- [ ] Feature Engineering with fe.py
- [ ] Train encoder
- [ ] Use encoder to generate more data
- [ ] Train model with CVTuner
- [ ] Make predictions

