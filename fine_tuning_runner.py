"""
This script automates the fine-tuning of a pre-trained model using labeled data and integrates with Google Cloud Platform (GCP).
 
It performs the following steps:

1. Downloads fine-tuning data from a specified GCP bucket.
2. Fine-tunes the pre-trained model using the downloaded data.
3. Optionally zips the fine-tuned model and uploads it back to a GCP bucket.

Key Features:
- Logs all operations for better traceability.
- Utilizes GCP utilities for data management.
- Modular design for easy customization.

Note:
- Ensure the Google Cloud Service Account credentials JSON file is available and its path is correctly set.
- Replace placeholder paths with actual paths to directories and files before running the script.
"""

import sys
import os
sys.path.append('/Users/navya/Desktop/Kounsel/MAIN') #Add Project Path to System Path. This ensures Python can locate the custom modules like FineTuneDataHandler, ZipUtils, and others present in the project directory.

import os
import logging
from com.mhire.fine_tuning.fine_tuning import ModelFineTune
# from com.mhire.utility.gcp_utils import GCPUtils
from com.mhire.utility.zip_util import ZipUtils

# # Configure logging
# logging.basicConfig(
#     filename="tmp/logs/logs.txt",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to Google Cloud Service Account credentials
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "PATH TO GCP SERVICE ACCOUNT CREDENTIALS JSON FILE"

def main():
    try:
        # Define constants
        BUCKET_NAME = "GCP BUCKET NAME"
        PRETRAINED_MODEL_DIR = "tmp/trained_model/"  # Pre-trained model local directory
        FINETUNED_MODEL_DIR = "tmp/fine_tuned_model/"  # Directory to save fine-tuned model
        FINETUNED_MODEL_ZIP = "tmp"  # Path to save zipped fine-tuned model

        # GCP paths
        REMOTE_FINETUNING_PATH = "PATH TO FINE-TUNING DATA IN GCP BUCKET"
        LOCAL_FINETUNING_PATH = "tmp/datasets/fine_tune_format.jsonl"  # Local path for downloaded data
        GCP_FINETUNED_MODEL_PATH = "PATH TO UPLOAD ZIPPED FINETUNED MODEL IN GCP BUCKET"

        # Step 1: Initialize GCP Utils
        # logging.info("Initializing GCP Utils")
        # gcp_utils = GCPUtils(BUCKET_NAME)

        # Step 2: Download fine-tuning data from GCP
        # logging.info(f"Downloading fine-tuning data from {REMOTE_FINETUNING_PATH}")
        # gcp_utils.download_file(REMOTE_FINETUNING_PATH, LOCAL_FINETUNING_PATH)

        # Step 3: Initializing fine-tuner
        logging.info("Initializing fine-tuner...")
        fine_tuner = ModelFineTune(PRETRAINED_MODEL_DIR, FINETUNED_MODEL_DIR, LOCAL_FINETUNING_PATH) #Create an instance of ModelFineTune and set paths for pre-trained model, fine-tuned model, and fine-tuning data.
        logging.info("Initialized fine-tuner successfully.")


        # Step 4: Fine-tune the model (Run fine-tuning)
        logging.info("Starting fine-tuning process...")
        fine_tuner.fine_tune()
        logging.info("Fine-tuning completed successfully.")

        # # Step 5: Zip the fine-tuned model
        # logging.info(f"Zipping fine-tuned model at {FINETUNED_MODEL_DIR}")
        # ZipUtils.zip_folder(FINETUNED_MODEL_DIR, FINETUNED_MODEL_ZIP)
        # logging.info(f"Model zipped successfully to {FINETUNED_MODEL_ZIP}")

        # # Step 6: Upload zipped model to GCP
        # logging.info(f"Uploading zipped model to {GCP_FINETUNED_MODEL_PATH}")
        # gcp_utils.upload_file(FINETUNED_MODEL_ZIP, GCP_FINETUNED_MODEL_PATH)
        # logging.info("Zipped model uploaded successfully to GCP")
        
    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
