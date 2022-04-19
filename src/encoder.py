from fe import get_train_data
from keras.callbacks import EarlyStopping
import utils

# Load
TRAINING = True
USE_FINETUNE = True
FOLDS = 5
SEED = 42

train, features, target = get_train_data()
train = utils.reduce_mem_usage(train)


X = train[features].values
y = train[target].values

# encode
autoencoder, encoder = utils.create_autoencoder(
    X.shape[-1], y.shape[-1], noise=0.1)
if TRAINING:
    autoencoder.fit(X, (X, y),
                    epochs=1000,
                    batch_size=4096,
                    validation_split=0.1,
                    callbacks=[EarlyStopping('val_loss', patience=10, restore_best_weights=True)])
    encoder.save_weights('./encoder.hdf5')
