
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, AlphaDropout
from tensorflow.keras.optimizers import Adam, Nadam # Changed from tensorflow.keras.optimizers.legacy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, RegressorMixin

def build_nn_v2(input_shape, units1=256, units2=128, units3=64, dropout_rate=0.3, use_alpha_dropout=False,
             activation1='relu', activation2='elu', activation3='tanh'):
    inputs = Input(shape=input_shape)
    DropoutLayer = AlphaDropout if use_alpha_dropout and activation1 == 'selu' else Dropout # Choose dropout layer based on activation

    # Layer 1
    x = Dense(units1, activation=activation1, kernel_initializer='he_normal' if activation1 != 'selu' else 'lecun_normal')(inputs)
    x = BatchNormalization()(x)
    x = DropoutLayer(dropout_rate)(x)
    
    # Layer 2
    x = Dense(units2, activation=activation2, kernel_initializer='he_normal' if activation2 != 'selu' else 'lecun_normal')(x)
    x = BatchNormalization()(x)
    x = DropoutLayer(dropout_rate)(x)
    
    # Layer 3
    x = Dense(units3, activation=activation3, kernel_initializer='he_normal' if activation3 != 'selu' else 'lecun_normal')(x)
    x = BatchNormalization()(x)
    # No dropout typically after the last hidden layer before the output layer
    
    outputs = Dense(1, activation='linear')(x) # Linear activation for regression
    model = Model(inputs=inputs, outputs=outputs)
    return model

class KerasRegressorWrapperV2(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn=build_nn_v2, input_shape=None, epochs=100, batch_size=32, lr=1e-3,
                 dropout_rate=0.3, units1=128, units2=64, units3=32, use_alpha_dropout=False,
                 activation1='relu', activation2='relu', activation3='relu', optimizer_name='Adam'):
        self.build_fn = build_fn
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.use_alpha_dropout = use_alpha_dropout # This will be True if SELU is chosen by Optuna and passed
        self.activation1 = activation1
        self.activation2 = activation2
        self.activation3 = activation3
        self.optimizer_name = optimizer_name
        self.model_ = None # scikit-learn convention for fitted attributes

    def fit(self, X, y, **fit_params):
        if self.input_shape is None and X is not None: # Ensure X is not None before accessing shape
            self.input_shape = (X.shape[1],)
        
        # Pass use_alpha_dropout to build_fn
        self.model_ = self.build_fn(
            self.input_shape, units1=self.units1, units2=self.units2, units3=self.units3, 
            dropout_rate=self.dropout_rate, use_alpha_dropout=self.use_alpha_dropout, 
            activation1=self.activation1, activation2=self.activation2, activation3=self.activation3
        )
        
        if self.optimizer_name == 'Adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer_name == 'Nadam':
            optimizer = Nadam(learning_rate=self.lr)
        else: # Default to Adam
            optimizer = Adam(learning_rate=self.lr)

        self.model_.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=0)
        ]
        
        val_data = fit_params.pop('validation_data', None)

        if val_data:
            self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, 
                            validation_data=val_data,
                            callbacks=callbacks, verbose=0, **fit_params)
        else:
            # If no validation_data is passed (e.g., by StackingRegressor), use validation_split
            self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, 
                            validation_split=0.15, # Creates a validation set from X, y
                            callbacks=callbacks, verbose=0, **fit_params)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been trained yet. Call fit first.")
        return self.model_.predict(X, batch_size=self.batch_size, verbose=0).flatten()

    def get_params(self, deep=True):
        # Ensure all __init__ params are listed here
        return {
            'build_fn': self.build_fn, 'input_shape': self.input_shape, 
            'epochs': self.epochs, 'batch_size': self.batch_size, 'lr': self.lr,
            'dropout_rate': self.dropout_rate, 'units1': self.units1, 'units2': self.units2, 
            'units3': self.units3, 'use_alpha_dropout': self.use_alpha_dropout, 
            'activation1': self.activation1, 'activation2': self.activation2, 
            'activation3': self.activation3, 'optimizer_name': self.optimizer_name
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
