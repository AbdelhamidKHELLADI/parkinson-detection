import click
import numpy as np
from utils.svm_utils import load_svm_model, predict_with_svm
from utils.preprocess_utils import process_folder
from utils.data_utils import load_test_features, segment_input
from dotenv import load_dotenv
import os

load_dotenv()

def run_prediction(input_audio_path):
    # Segment audio & extract features
    segment_input(input_audio_path)
    process_folder(os.getenv("TMP_PATH"),test=True)  

    # Load model
    clf = load_svm_model(os.getenv("MODEL_PATH"))
    features = load_test_features(os.getenv("TMP_FEATURES_PATH"))

    # Predict
    predictions = predict_with_svm(clf, features)
    result = "healthy" if np.median(predictions) == 0 else "parkinson"

    return result


@click.command()
@click.argument("audio", type=click.Path(exists=True))
def main(audio):
    """Run Parkinson/Healthy prediction on an input AUDIO file."""
    result = run_prediction(audio)
    click.echo(result) 


if __name__ == "__main__":
    main()
