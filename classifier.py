import sys
import os
sys.path.append('/Users/navya/Desktop/Kounsel/MAIN') #Add Project Path to System Path. This ensures Python can locate the custom modules like FineTuneDataHandler, ZipUtils, and others present in the project directory.

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
import logging
import json

# Setup logging for debugging and information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Classifier(Dataset):
    """
    Custom Dataset to handle tokenized text inputs for recipe data.
        This class prepares the data for the model:
            Inputs: Tokenized data (input IDs, attention masks, and labels).
            Outputs: Returns batches of data to the model during evaluation.
    """
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    @staticmethod
    def load_ingredient_lists(vegan_file, vegetarian_file, non_veg_file):
        """
        Load ingredient lists from JSONL files for vegan, vegetarian, and non-vegetarian categories.
        """
        try:
            with open(vegan_file, 'r') as f:
                vegan_ingredients = [json.loads(line)['ingredient'] for line in f]
            with open(vegetarian_file, 'r') as f:
                vegetarian_ingredients = [json.loads(line)['ingredient'] for line in f]
            with open(non_veg_file, 'r') as f:
                non_veg_ingredients = [json.loads(line)['ingredient'] for line in f]
            logging.info("Ingredient lists loaded successfully.")
            return vegan_ingredients, vegetarian_ingredients, non_veg_ingredients
        except Exception as e:
            logging.error(f"Error loading ingredient lists: {e}")
            raise

def preprocess_testing_data(data):
    """
    Preprocess testing data by extracting relevant fields (ingredients, instructions, toppings).
    """
    logging.info("Preprocessing the testing data from tester script...")
    logging.info(f"First few entries of test data: {test_data[:5]}")

    try:
        processed = {
            'text': [],
            'label': []  # Optional, for evaluation purposes
        }

        for entry in data:
            # Initialize text as an empty string
            text = ""

            # Check if 'information' exists in entry
            if 'information' in entry:
                # Case 1: 'information' is a string (simple case)
                if isinstance(entry['information'], str):
                    text = entry['information']

                # Case 2: 'information' is a dictionary (contains diet suggestions or similar keys)
                elif isinstance(entry['information'], dict):
                    # Extract diet suggestions and other keys
                    for key, value in entry['information'].items():
                        if isinstance(value, list):
                            # Join the list items into a sentence
                            text += f"{key}: {', '.join(value)}. "
                        elif isinstance(value, str):
                            # Combine key and string value into a sentence
                            text += f"{key}: {value}. "
                else:
                    # In case 'information' is neither a string nor a dictionary
                    logging.warning(f"Unexpected format for entry: {entry}")
                    continue  # Skip this entry
            else:
                logging.warning(f"No 'information' key found for entry: {entry}")
                continue  # Skip this entry if 'information' is missing

            processed['text'].append(text.strip())  # Ensure no trailing spaces
            processed['label'].append(entry.get('label', 2))  # Default label to 2 (non-veg) if not present

        logging.info(f"Processed {len(processed['text'])} samples.")
        return processed

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def tokenize_data(processed_data, tokenizer, max_length=512):
        """
        Tokenize the processed data using the tokenizer.
        This function tokenizes the text data:
            Uses the BERT tokenizer to convert text into input IDs and attention masks.
            Pads or truncates the text to a fixed length (max_length=512).
            Converts labels into tensors.
        """
        logging.info("Tokenizing the data from tester script...")
        try:
            tokenized_data = {
                'input_ids': [],
                'attention_mask': [],
                'labels': processed_data['label']
            }

            for text in processed_data['text']:
                encoding = tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                tokenized_data['input_ids'].append(encoding['input_ids'].squeeze(0))
                tokenized_data['attention_mask'].append(encoding['attention_mask'].squeeze(0))

            tokenized_data['input_ids'] = torch.stack(tokenized_data['input_ids'])
            tokenized_data['attention_mask'] = torch.stack(tokenized_data['attention_mask'])
            tokenized_data['labels'] = torch.tensor(tokenized_data['labels'])

            logging.info("Tokenization complete.")
            return tokenized_data

        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
            raise

def classify_recipe(data, tokenizer, vegan_ingredients, vegetarian_ingredients, non_veg_ingredients):
    """
    Classify the recipe as vegan, vegetarian, or non-vegetarian based on its components.
    """
    try:
        # Preprocess the recipe data
        processed_data = preprocess_testing_data(data)
        text = processed_data['text'][0]  # Extract the processed text for the single recipe

        # Tokenize the text
        tokenized_data = tokenize_data(processed_data, tokenizer)

        # Use text for classification logic
        recipe_ingredients = text.split()  # Split the text into words for classification

        # Apply the ingredient classification logic
        if any(meat in recipe_ingredients for meat in non_veg_ingredients):
            return 2  # Non-Veg
        elif any(dairy in recipe_ingredients for dairy in vegetarian_ingredients):
            return 1  # Veg
        elif all(ingredient in vegan_ingredients for ingredient in recipe_ingredients):
            return 0  # Vegan
        else:
            return -1  # Undefined or error case
    except Exception as e:
        logging.error(f"Error classifying recipe: {e}")
        raise

def evaluate_model(model, tokenizer, test_data, vegan_ingredients, vegetarian_ingredients, non_veg_ingredients, batch_size=8):
    """
    Evaluate the fine-tuned model on the test dataset and calculate accuracy.
    """
    logging.info("Starting evaluation (evaluate model function)...")
    try:
        # Preprocess the recipe data
        processed_data = preprocess_testing_data(test_data)
        logging.info("Preprocessing of test data completed successfully.")

        # Extract the processed text for the recipes
        recipe_texts = processed_data['text']

        # Tokenize the text
        tokenized_data = tokenize_data(processed_data, tokenizer)
        logging.info("Tokenization of test data completed successfully.")

        # Create dataset and dataloader
        dataset = Classifier(tokenized_data)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                # Use classify_recipe to classify each recipe
                batch_start_idx = idx * batch_size
                for i, recipe_text in enumerate(recipe_texts[batch_start_idx:batch_start_idx + batch_size]):
                    # Classify each recipe based on its ingredients
                    predicted_label = classify_recipe(recipe_text, tokenizer, vegan_ingredients, vegetarian_ingredients, non_veg_ingredients)
                    all_predictions.append(predicted_label)
                    all_labels.append(labels[batch_start_idx + i].cpu().numpy())

        # Compute accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        logging.info(f"Evaluation complete: Accuracy={accuracy:.4f}")

        return {'accuracy': accuracy}

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Path to the directory where the fine-tuned model is saved
    MODEL_DIR = "tmp/model"
    test_data = [
        {"information": "Saffron Milk. Ingredients: 1 cup milk, 1/4 tsp saffron strands, 1 tbsp sugar. Instructions: Boil milk with saffron and sugar. Stir and serve hot.", "label": 1},
        {"information": {"Diet suggestion for Heart Disease patients": ["Choose lean meats such as chicken and turkey.", "Incorporate fatty fish like salmon for omega-3 fatty acids.", "Limit saturated fats and trans fats by avoiding processed meats and fried foods."]}, "label": 2, "context": "A heart-healthy diet includes lean meats and fish rich in omega-3s, while limiting unhealthy fats to promote cardiovascular health."},
        {"information": "Breakfast: Scrambled eggs with spinach, Lunch: Beef and vegetable stir-fry, Dinner: Roasted duck with wild rice", "label": 2, "context": "This meal plan features eggs, beef, and duck, all non-vegetarian ingredients."},
        {"information": {"Diet suggestion for heart disease patients": ["Include heart-healthy fats like olive oil, avocados, and nuts, along with fiber-rich vegetables and fruits.", "Avoid saturated fats, processed foods, and excessive salt."]}, "label": 0, "context": "This heart-healthy diet helps reduce inflammation and manage cholesterol levels by focusing on nutrient-dense, anti-inflammatory foods."},
        {"information": "Vegan Chili. Ingredients: 1 cup kidney beans (canned or cooked), 1 cup black beans (canned or cooked), 1/2 cup diced tomatoes, 1/4 cup chopped onion, 1 tbsp chili powder, 1 tbsp cumin, 1 tbsp olive oil. Instructions: Heat olive oil in a pot. Sauté onion until soft, then add tomatoes, beans, chili powder, cumin, and salt. Simmer for 30 minutes. Serve hot.", "label": 0, "context": "This chili is plant-based, made with kidney beans, black beans, and spices, all plant-derived ingredients."}

    ]   

    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

        # Define ingredient lists
        vegan_ingredients = ["tofu", "broccoli", "carrot", "lentils"]
        vegetarian_ingredients = ["milk", "cheese", "yogurt"]
        non_veg_ingredients = ["chicken", "beef", "fish"]

        metrics = evaluate_model(model, tokenizer, test_data, vegan_ingredients, vegetarian_ingredients, non_veg_ingredients, batch_size=4)
        print(metrics)

    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")
    
    # # TEST_DATA_FILE = "fine_tune_format.jsonl"  # Path to the JSONL file
    # TEST_DATA_FILE = "FoodClassificationLLM_test.jsonl"  # Path to the JSONL file

    # try:
    #     tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    #     model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

    #     # Load test data from JSONL file
    #     logging.info(f"Loading test data from {TEST_DATA_FILE}...")
    #     with open(TEST_DATA_FILE, 'r') as f:
    #         test_data = [json.loads(line) for line in f]

    #     metrics = evaluate_model(model, tokenizer, test_data, batch_size=4)
    #     print(metrics)

    # except Exception as e:
    #     logging.error(f"Critical error in main execution: {e}")

