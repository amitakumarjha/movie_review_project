import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import data_processing as dp
import movie_review_model.preProcessing.encoder as en

def build_transformer_encoder():
    # Build the Transformer encoder
    vocab_size = 20000
    embed_dim = 256
    num_heads = 2
    dense_dim = 32

    inputs = keras.Input(shape=(1,), dtype=tf.string)
    x = dp.text_vectorization(inputs)                                         # TextVectorization layer
    x = layers.Embedding(vocab_size, embed_dim)(x)                         # Embedding layer
    x = en.TransformerEncoder(embed_dim, dense_dim, num_heads)(x)             # Transformer Encoder block
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)                     # Dense layer for classification

    model = keras.Model(inputs, outputs)

    model.compile(optimizer="rmsprop",
                loss="binary_crossentropy",
                metrics=["accuracy"])

    return model