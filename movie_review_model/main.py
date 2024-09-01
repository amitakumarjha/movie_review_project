import movie_review_model.prediction.make_prediction as mp
import movie_review_model.model_training.evaluation as eva

def prediction(review_input):
    model = eva.load_model()
    result = mp.make_prediction(review_input, model)
    print(result)
    return result

if __name__ == "__main__":
    prediction("You won't get bored for even a second in the entire movie due to it being full of comedy with perfect punch timing of the actors. It is a good blend of less horror and more comedy")