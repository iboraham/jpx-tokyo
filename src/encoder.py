from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import numpy as np
import pandas as pd
import datatable as dt
from sklearn.model_selection import GroupKFold

from tqdm import tqdm
from random import choices

import kerastuner as kt

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

from utils import *

# Load
TRAINING = True
USE_FINETUNE = True
FOLDS = 5
SEED = 42

train = dt.fread(
    '../db/train_files/stock_prices.csv').to_pandas()
train = prep_prices(train)
train = reduce_mem_usage(train)

features = ["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume"]

X = train[features].values
y = train['Target'].values  # Multitarget

# encode
autoencoder, encoder = create_autoencoder(X.shape[-1], y.shape[-1], noise=0.1)
if TRAINING:
    autoencoder.fit(X, (X, y),
                    epochs=1000,
                    batch_size=4096,
                    validation_split=0.1,
                    callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True)])
    encoder.save_weights('./encoder.hdf5')
