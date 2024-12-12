import tensorflow as tf

def load_model(model_path="my_model"):
    """
    Load the saved TensorFlow model.
    """
    return tf.keras.models.load_model(model_path)

def predict(model, input_data):
    """
    Make a prediction with the given model.
    """
    return model.predict(input_data)
