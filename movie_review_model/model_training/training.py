import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from movie_review_model.preProcessing.split_data import text_dataset
import movie_review_model.preProcessing.transformer_encoder_model as tem

def model_training():

    train_ds, val_ds, test_ds = text_dataset()

    # Fit the model on train set
    callbacks = [keras.callbacks.ModelCheckpoint("trained_models/transformer_encoder.keras", save_best_only=True)]

    # Change target shape from (None,) to (None, 1)
    train_dataset = text_dataset().train_ds.map(lambda x, y: (x, tf.reshape(y, (-1,1))))
    val_dataset = val_ds.map(lambda x, y: (x, tf.reshape(y, (-1,1))))

    model = tem.build_transformer_encoder()

    model.fit(train_dataset,
            validation_data = val_dataset,
            epochs = 10,
            callbacks = callbacks)