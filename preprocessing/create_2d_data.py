import os
import glob
import librosa
from utils.preprocess_utils import create_spectrogam, augment_spectrogram, save_spectrogram
from dotenv import load_dotenv

load_dotenv()



if __name__ == "__main__":

    destination = os.getenv("SPECTROGRAM_PATH")  # Path to save spectrograms
    sr = int(os.getenv("TARGET_SAMPLE_RATE"))
    database="1AS" #1AS, 1FS, 5AS, 5FS
    path = os.getenv("SEGMENTS_PATH") # Path to the folder containing audio files
    dest_dir = os.path.join(destination, database)
    os.makedirs(dest_dir, exist_ok=True)
    
    
    for category in os.listdir(os.path.join(path, database)):
        dest_sub_dir = os.path.join(dest_dir, category)
        os.makedirs(dest_sub_dir, exist_ok=True)
        
        for record in glob.glob(os.path.join(path, database, category) + "/*.wav"):
            print(record)
            output_file = os.path.join(dest_sub_dir, os.path.basename(record).replace('.wav', '_original.png'))
            audio, _ = librosa.load(record, sr=sr)
            original_spec=create_spectrogam(audio, sr)
            specs = augment_spectrogram(audio)
            time_masked_spec = specs["time_masked"]
            freq_masked_spec = specs["freq_masked"]
            combined_masked_spec = specs["combined"]
        
            save_spectrogram(original_spec, output_file, sr)
            save_spectrogram(time_masked_spec, output_file.replace('_original.png', '_time_masked.png'), sr)
            save_spectrogram(freq_masked_spec, output_file.replace('_original.png', '_freq_masked.png'), sr)
            save_spectrogram(combined_masked_spec, output_file.replace('_original.png', '_combined_masked.png'), sr)


            
