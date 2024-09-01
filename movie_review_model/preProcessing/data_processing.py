import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import split_data as sd

def TextVectorization():
    # Vectorizing the data
    max_length = 600
    max_tokens = 20000
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,    # Q: What is the vocabular size?
        output_mode="int",        # Q: What will be the type of output for a token (say), 'amazing' ?
        output_sequence_length=max_length,      # Q: What is the maximum length of review? Is it a fair assumption?
        )

    text_only_train_ds = sd.text_only_train()
    text_vectorization.adapt(text_only_train_ds)

# Custom self attention function
def self_attention(input_sequence):
    output = np.zeros(shape=input_sequence.shape)
    for i, pivot_vector in enumerate(input_sequence): # iterate over each token in ip seq
        scores = np.zeros(shape=(len(input_sequence), ))

        for j, vector in enumerate(input_sequence):
            scores[j] = np.dot(pivot_vector, vector.T)    # Pairwise scores

        scores /= np.sqrt(input_sequence.shape[1]) # scale #[1] is the embedding dim
        scores = tf.nn.softmax(scores)              # softmax
        new_pivot_representation = np.zeros(shape=pivot_vector.shape)
        for j, vector in enumerate(input_sequence):
            new_pivot_representation += vector*scores[j] # weigthed sum
        output[i] = new_pivot_representation
    return output