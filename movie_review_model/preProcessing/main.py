import movie_review_model.preProcessing.data_processing as dp
import movie_review_model.preProcessing.split_data as sd

if __name__ == "__main__":
    
    sd.train_val_split_data()
    sd.text_dataset()
    dp.TextVectorization()