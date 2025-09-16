from utils.preprocess_utils import process_folder
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    database="1AS" #1AS, 1FS, 5AS, 5FS
    folder = os.path.join(os.getenv("SEGMENTS_PATH"), database)  # Path to the folder containing audio files
    process_folder(folder)
