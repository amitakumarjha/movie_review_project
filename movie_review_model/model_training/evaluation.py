from tensorflow import keras
import movie_review_model.preProcessing.encoder as en
import movie_review_model.preProcessing.split_data as sd

import movie_review_model
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_model():
    
    PACKAGE_ROOT = Path(movie_review_model.__file__).resolve().parent
    trained_model_path = PACKAGE_ROOT / "trained_models/transformer_encoder.keras"

    model = keras.models.load_model(
    trained_model_path,
    custom_objects={"TransformerEncoder": en.TransformerEncoder}
    )
    return model

def model_evaluation():
    train_ds, val_ds, test_ds = sd.text_dataset()
    model = load_model
    return model.evaluate(test_ds)[1]
    #print(f"Test acc: {model.evaluate(test_ds)[1]:.3f}")