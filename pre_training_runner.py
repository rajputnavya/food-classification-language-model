import sys
import os
sys.path.append('/Users/navya/Desktop/Kounsel/MAIN') #

from transformers import logging as hf_logging

# import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
# from transformers.utils import logging
from com.mhire.data_processing.pre_training_data_handler import PreTrainingDataHandler
from com.mhire.pre_training.pre_training import Pretraining

# Set up logging
# logger = logging.get_logger("transformers.trainer")
# logger.setLevel(logging.INFO)
hf_logging.set_verbosity_info()  # This sets the verbosity to INFO level
logger = hf_logging.get_logger()

def run_pretraining():
    try:
        # Set up directories
        LOCAL_DIR = "tmp/datasets/"
        OUTPUT_DIR = "tmp/trained_model/"
        LOG_DIR = "tmp/logs/"
        
        CLEAN_DATA_FILE = os.path.join(LOCAL_DIR, "mlm_format.jsonl")
        NSP_FORMAT_FILE = os.path.join(LOCAL_DIR, "nsp_format.jsonl")

        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        try:
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", model_max_length=512)
            logger.info("Tokenizer successfully initialized.")
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise

        # Initialize DataHandler and prepare datasets
        logger.info("Initializing DataHandler and preparing datasets...")
        try:
            data_handler = PreTrainingDataHandler(tokenizer)
            nsp_dataset = data_handler.prepare_nsp_dataset(file_path=NSP_FORMAT_FILE)
            mlm_dataset = data_handler.prepare_mlm_dataset(file_path=CLEAN_DATA_FILE)
            combined_dataset = data_handler.combine_datasets(mlm_dataset, nsp_dataset)
            logger.info("DataHandler initialization and dataset preparation completed.")
        except Exception as e:
            logger.error(f"Error initializing DataHandler or preparing datasets: {e}")
            raise

        # Split datasets into train and validation
        logger.info("Splitting datasets into train and validation...")
        try:
            train_indices, val_indices = train_test_split(
                range(len(combined_dataset)), test_size=0.2, random_state=42
            )
            train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)
            logger.info("Datasets split into train and validation completed.")
        except Exception as e:
            logger.error(f"Error splitting datasets: {e}")
            raise

        # Initialize pretraining and train model
        logger.info("Initializing pretraining and starting model training...")
        try:
            pretrainer = Pretraining("bert-base-uncased", OUTPUT_DIR, LOG_DIR, tokenizer)
            pretrainer.train(train_dataset, val_dataset)
            logger.info("Model successfully trained.")
        except Exception as e:
            logger.error(f"Error during pretraining or model training: {e}")
            raise
        
    except Exception as e:
        logger.critical(f"Pretraining failed: {e}")
        raise

if __name__ == "__main__":
    run_pretraining()
