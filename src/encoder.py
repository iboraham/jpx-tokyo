from fe import get_train_data
from keras.callbacks import EarlyStopping
import utils

# Load
TRAINING = True
USE_FINETUNE = True
FOLDS = 5
SEED = 42

train, features, target = get_train_data()


X = train[features].values
y = train[target].values.reshape(-1, 1)

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
