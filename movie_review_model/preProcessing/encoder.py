import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim    # Dimension of embedding. 4 in the dummy example
        self.dense_dim = dense_dim    # No. of neurons in dense layer
        self.num_heads = num_heads    # No. of heads for MultiHead Attention layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)    # MultiHead Attention layer
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"),
                                            layers.Dense(embed_dim),]    # encoders are stacked on top of the other.
                                           )                             # So output dimension is also embed_dim
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    # Call function based on figure above
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(query=inputs,             # Query: inputs,
                                          value=inputs,             # Value: inputs,
                                          key=inputs,               # Keys: Same as Values by default
                                          attention_mask=mask
                                          )                         # Q: Can you see how this is self attention? A: all args are the same

        proj_input = self.layernorm_1(inputs + attention_output) # LayerNormalization; + Recall cat picture
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)  # LayerNormalization + Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config