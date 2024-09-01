import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras.utils import text_dataset_from_directory

def train_val_split_data():
    # move 20% of the training data to the validation folder
    base_dir = pathlib.Path("../dataset/aclImdb")
    val_dir = base_dir / "val"
    train_dir = base_dir / "train"
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        # random.Random(1337).shuffle(files) # We should shuffle. Only commenting for demonstration
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname,
                        val_dir / category / fname)
            
def text_dataset():
    # Create dataset using utility
    batch_size = 32
    train_ds = text_dataset_from_directory("../dataset/aclImdb/train", batch_size=batch_size)
    val_ds = text_dataset_from_directory("../dataset/aclImdb/val", batch_size=batch_size)
    test_ds = text_dataset_from_directory("../dataset/aclImdb/test", batch_size=batch_size)
    
    return train_ds, val_ds, test_ds

def text_only_train():

    train_ds, val_ds, test_ds = text_dataset()

    # Extracting only the review text(not labels); to be used later to adapt the TextVec layer
    text_only_train_ds = train_ds.map(lambda x, y: x)   
    
    return  text_only_train_ds         # lambda x, y: x  --> replace x,y with x. That is remove labels, just keep text data.