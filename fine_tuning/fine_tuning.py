import sys
import os
sys.path.append('/Users/navya/Desktop/Kounsel/MAIN') #

import os
import logging
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, EarlyStoppingCallback
from com.mhire.data_processing.fine_tune_data_handler import FineTuneDataHandler
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from com.mhire.utility.zip_util import ZipUtils
from sklearn.metrics import classification_report

# # Configure logging
# logging.basicConfig(
#     filename='temp/logs/logs.txt',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelFineTune:
    """
    Handles the fine-tuning of a pre-trained model using labeled JSONL data.

    Attributes:
        PRETRAIN_MODEL_DIR (str): Pre-trained model directory.
        LOCAL_MODEL_DIR (str): Local directory to save the fine-tuned model.
        training_file_path: Fine-tuning data
    """

    def __init__(self, PRETRAIN_MODEL_DIR, LOCAL_MODEL_DIR, training_file_path):
        self.PRETRAIN_MODEL_DIR = PRETRAIN_MODEL_DIR
        self.LOCAL_MODEL_DIR = LOCAL_MODEL_DIR
        self.training_file_path = training_file_path
        self.tokenizer = BertTokenizer.from_pretrained(self.PRETRAIN_MODEL_DIR)  # Initialize the tokenizer

    def fine_tune(self):
        try:
            # Step 1: Load and preprocess fine-tuning data
            logging.info("Step-1: Load and preprocess fine-tuning data...")
            finetuning_data = FineTuneDataHandler.load_finetuning_data(self.training_file_path)
            processed_data = FineTuneDataHandler.preprocess_data(finetuning_data) #processed data
            logging.info("Step-1: Load and preprocess fine-tuning data completed successfully.")

            # Step 2: Tokenize the processed data
            logging.info("Step-2a: Tokenizing the dataset...")
            tokenized_data = FineTuneDataHandler.tokenize_data(processed_data, self.tokenizer)
            logging.info("Step-2a: Tokenizing the dataset completed successfully.")

            # Convert tokenized data to Hugging Face dataset format
            logging.info('Step-2b: Convert tokenized data to Hugging Face dataset format...')
            dataset = Dataset.from_dict(tokenized_data)
            logging.info('Step-2b: Convert tokenized data to Hugging Face dataset format completed successfully.')

            # Step 3: Split the dataset using Hugging Face's `train_test_split`
            # Training set: 70%, Validation set: 15%, Test set: 15%
            logging.info('Step-3: Splitting dataset into train, validation, and test sets...')
            train_test = dataset.train_test_split(test_size=0.3, seed=42)
            train_data = train_test['train']
            test_val = train_test['test']
            val_test = test_val.train_test_split(test_size=0.5, seed=42)
            val_data = val_test['train']
            test_data = val_test['test']

            logging.info('Step-3: Splitting dataset into train, validation, and test sets completed successfully.')
            logging.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

            # Step 4: Define the model (Load Pre-trained Model):
            logging.info('Step-4: Define the model (Load Pre-trained Model)...')
            model = BertForSequenceClassification.from_pretrained(self.PRETRAIN_MODEL_DIR, num_labels=3) ### 3 labels for vegan, veg and non-veg (model's output layer has 3 neurons)
            logging.info('Step-4: Define the model (Load Pre-trained Model) completed successfully.')

            # Step 5: Set training arguments
            logging.info('Step-5: Set training arguments...')
            training_args = TrainingArguments(
                                output_dir= self.LOCAL_MODEL_DIR,  # Intermediate checkpoints are saved here
                                overwrite_output_dir=True, 
                                num_train_epochs=5,
                                per_device_train_batch_size=16,
                                per_device_eval_batch_size=64,
                                evaluation_strategy="epoch",  # Evaluate at the end of each epoch
                                save_strategy="epoch",        # Save the model at the end of each epoch
                                logging_dir='tmp/logs',
                                logging_steps=10,
                                load_best_model_at_end=True,  # Load the best model at the end of training
                                metric_for_best_model='accuracy',
                                report_to="tensorboard",
                                gradient_accumulation_steps=2,
                            )
            logging.info('Step-5: Set training arguments completed successfully.')

            # Step 6: Define metrics for evaluation
            logging.info('Step-6: Define metrics for evaluation...')
            # def compute_metrics(p):
            #     predictions, labels = p
            #     preds = predictions.argmax(axis=1) ### Get predicted class indices

            #     accuracy = accuracy_score(labels, preds)
            #     precision = precision_score(labels, preds)
            #     recall = recall_score(labels, preds)
            #     f1 = f1_score(labels, preds)

            #     return {
            #         'accuracy': accuracy,
            #         'precision': precision,
            #         'recall': recall,
            #         'f1': f1,
            #     }

            def compute_metrics(p):
                predictions, labels = p
                preds = predictions.argmax(axis=1)  # Get predicted class indices
                
                report = classification_report(labels, preds, target_names=["Vegan", "Vegetarian", "Non-Vegetarian"], output_dict=True)
                return {
                    "accuracy": report["accuracy"],
                    "precision": report["macro avg"]["precision"],
                    "recall": report["macro avg"]["recall"],
                    "f1": report["macro avg"]["f1-score"],
                }
            logging.info('Step-6: Define metrics for evaluation completed successfully.')

            # Step 7: Set up the Trainer
            logging.info('Step-7: Setting up the Trainer starting...')
            trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_data,
                        eval_dataset=val_data,
                        compute_metrics=compute_metrics,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Stop if no improvement for 2 evals

                    )
            logging.info('Step-7: Set up the Trainer completed successfully.')

            # Step 8: Start training the model
            logging.info('Step-8: Starting model training...')
            trainer.train()
            logging.info('Step-8: Model training completed successfully.')

            # Step 9: Save the fine-tuned model locally (Fine_tuned_model directory)
            logging.info('Step-9: Saving the fine-tuned model...')
            os.makedirs(self.LOCAL_MODEL_DIR, exist_ok=True)
            trainer.save_model(self.LOCAL_MODEL_DIR) # copy the best model to the directory specified by self.LOCAL_MODEL_DIR.
            logging.info(f'Model saved locally at: {self.LOCAL_MODEL_DIR}')

            ###
            # Save tokenizer and training arguments
            self.tokenizer.save_pretrained(self.LOCAL_MODEL_DIR)
            logging.info(f"Tokenizer saved at: {self.LOCAL_MODEL_DIR}")

            with open(os.path.join(self.LOCAL_MODEL_DIR, 'training_args.json'), 'w') as f:
                f.write(training_args.to_json_string())
            logging.info("Training arguments saved successfully.")

            # # Step 10: Zip the fine-tuned model
            # FINETUNED_MODEL_ZIP = "fine_tuned_model.zip"
            # ZipUtils.zip_folder(self.LOCAL_MODEL_DIR, FINETUNED_MODEL_ZIP)
            # logging.info(f"Fine-tuned model zipped to: {FINETUNED_MODEL_ZIP}")

        except Exception as e:
            logging.error(f'Fine-tuning process failed: {str(e)}')
            raise
