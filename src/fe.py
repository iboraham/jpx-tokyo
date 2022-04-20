import pandas as pd
import datatable as dt
import numpy as np
PATH = 'db/train_files/stock_prices.csv'


def get_train_data() -> tuple[pd.DataFrame, list, str]:
    """
    Get the training data.

    Returns:
        tuple[pd.DataFrame, list, str]: The training data and the list of features.
    """
    df = read_data()
    df = feature_engineering(df)
    features = ['Side', 'ret_H', 'ret_L', 'ret', 'ret_Div',
                'log_Dollars', 'GK_sqrt_vol', 'RS_sqrt_vol']
    df[features] = df[features].astype('float32')
    return df, features, 'Target'


def read_data() -> pd.DataFrame:
    """
    Read the data.

    Returns:
        pd.DataFrame: The data.
    """
    df = dt.fread(f'{PATH}').to_pandas()
    return df

def prep_prices(prices):
    prices.Date = pd.to_datetime(prices.Date).view(int)
    prices["Volume"].fillna(1,inplace=True)
    prices.fillna(0,inplace=True)
    return prices

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering.

    Args:
        df (pd.DataFrame): The data.

    Returns:
        pd.DataFrame: The data with the new features.
    """
    df = prep_prices(df)
    df['Avg_Price'] = (df['Close']+df['Open'])/2
    df['Avg_Price_HL'] = (df['High']+df['Low'])/2
    df['Side'] = 2*(df['Avg_Price']-df['Avg_Price_HL']) / \
        (df['High']-df['Low']+1)

    df['ret_H'] = df['High']/(df['Open']+1)
    df['ret_L'] = df['Low']/(df['Open']+1)
    df['ret'] = df['Close']/(df['Open']+1)
    df['ret_Div'] = df['ExpectedDividend']/(df['Open']+1)

    df['log_Dollars'] = np.log(df['Avg_Price']*df['Volume'])

    df['GK_sqrt_vol'] = np.sqrt((1 / 2 * np.log(df['High']/(df['Low']+1)) ** 2 - (
        2 * np.log(2) - 1) * np.log(df['Close'] / (df['Open'])+1) ** 2))
    df['RS_sqrt_vol'] = np.sqrt(np.log(df['High']/(df['Close']+1))*np.log(df['High']/(
        df['Open']+1)) + np.log(df['Low']/(df['Close']+1))*np.log(df['Low']/(df['Open']+1)))
    return df
