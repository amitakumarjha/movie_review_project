import tensorflow as tf

def make_prediction(input_text, tf_model):
    myTensor = tf.convert_to_tensor(input_text, dtype=tf.string)
    pred = tf_model(tf.reshape(myTensor, (-1,1)))
    label_index = int(pred.numpy()[0,0] + 0.5)
    mapping = {0: 'Negative', 1: 'Positive'}
    label = mapping[label_index]
    return label