from dotenv import load_dotenv
from utils.segment_utils import process_healthy, process_parkinsons
import os
load_dotenv()

target_sample_rate = os.getenv("TARGET_SAMPLE_RATE")
n_fft = os.getenv("N_FFT")
hop_length = os.getenv("HOP_LENGTH")
targets = [1] # durations in seconds


def main():
    process_healthy(os.getenv("PATH1"))
    process_parkinsons(os.getenv("PATH2"))

if __name__ == "__main__":  
    main()
