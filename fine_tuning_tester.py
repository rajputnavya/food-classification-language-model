'''
Evaluation Pipeline:
1. Preprocess and tokenize the test data.
2. Use the model to predict labels for the test data.
3. Compare the predicted labels with the actual labels to compute accuracy or other metrics.
'''
import sys
import os
sys.path.append('/Users/navya/Desktop/Kounsel/MAIN') #

from com.mhire.data_processing.fine_tune_data_handler import FineTuneDataHandler
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
import logging
import json

# Configure logging to display messages on the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RecipeDataset(Dataset):
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

def preprocess_data(data): ### CALL FROM FINE TUNE DATA HANDLER
    """
    Preprocess testing data by extracting relevant fields (ingredients, instructions, toppings).
    This function preprocesses the raw test data:
        Extracts recipe components (ingredients, instructions, toppings) from the input JSON.
        Concatenates them into a single string.
        Handles cases where the data is missing or poorly formatted.
        Returns a dictionary of text (recipe descriptions) and corresponding labels.
    """
    logging.info("Preprocessing the testing data from tester script...")
    try:
        processed = {
            'text': [],
            'label': []  # Optional, for evaluation purposes
        }

        for entry in data:
            # Initialize text as an empty string
            text = ""

            # Case 1: 'information' is a string (simple case)
            if isinstance(entry.get('information'), str):
                text = entry['information']

           # Case 2: 'information' is a dictionary (known keys like 'ingredients', 'instructions', etc.)
            elif isinstance(entry.get('information'), dict):
                # Example: Handle 'ingredients', 'instructions', and 'toppings'
                if 'ingredients' in entry['information'] or 'instructions' in entry['information']:
                    ingredients = entry['information'].get('ingredients', [])
                    instructions = entry['information'].get('instructions', '')
                    toppings = entry['information'].get('toppings', [])

                    if ingredients:
                        text += f"ingredients: {', '.join(ingredients)}. "
                    if instructions:
                        text += f"instructions: {instructions}. "
                    if toppings:
                        text += f"toppings: {', '.join(toppings)}."

                # NEW: Handle descriptive keys like "Diet suggestion for Heart Disease patients"
                else:
                    for key, value in entry['information'].items():
                        if isinstance(value, list):
                            # Combine key and list items into a sentence
                            text += f"{key}: {', '.join(value)}. "
                        elif isinstance(value, str):
                            # Combine key and string value into a sentence
                            text += f"{key}: {value}. "
            
            processed['text'].append(text.strip()) # Ensure no trailing spaces
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

def evaluate_model(model, tokenizer, test_data, batch_size=8):
    """
    Evaluate the fine-tuned model on the test dataset and calculate metrics.
    This is the main function that evaluates the model:
        Step 1: Preprocess and tokenize the test data.
        Step 2: Load the tokenized data into a DataLoader for batch processing.
        Step 3: Move the model and data to the appropriate device (GPU or CPU).
        Step 4: Predict the labels using the model and compare them with the actual labels.
        Step 5: Compute metrics (accuracy, precision, recall, F1-score) and generate a classification report.
        Step 6: Log the results for debugging and analysis.
    """
    logging.info("Starting evaluation (evaluate model function)...")
    try:
        processed_data = preprocess_data(test_data)
        logging.info("Preprocessing of test data completed successfully.")

        tokenized_data = tokenize_data(processed_data, tokenizer)
        logging.info("Tokenization of test data completed successfully.")

        dataset = RecipeDataset(tokenized_data)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_labels = []
        all_predictions = []
        recipe_texts = processed_data['text']  # Keep track of the original recipes
        prediction_text_pairs = []  # To store recipe and prediction pairs

        with torch.no_grad():
            for idx, batch in enumerate(data_loader): ### idx defined
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

                # Pair recipes with predictions for display
                batch_start_idx = idx * batch_size
                for i, prediction in enumerate(predictions.cpu().numpy()):
                    recipe_text = recipe_texts[batch_start_idx + i]
                    prediction_text_pairs.append((recipe_text, prediction))

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        logging.info(f"Evaluation complete: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

        class_report = classification_report(all_labels, all_predictions, labels=[0, 1, 2], target_names=['vegan', 'vegetarian', 'non-vegetarian'])
        logging.info(f"Classification Report:\n{class_report}")

        # Log predictions with recipes
        logging.info("Predicted labels with corresponding recipes:")
        for recipe, pred in prediction_text_pairs:
            logging.info(f"Text: {recipe} -> Predicted Label: {pred}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Path to the directory where the fine-tuned model is saved
    MODEL_DIR = "model3"
    test_data = [
        {"information": "almond milk, tofu, vegan cream cheese, eggs, cheese, vegetarian meat, soy milk, apple, cow milk, ghee, butter", "label": 1},
        {"information": "almond milk, tofu, lentils, apple, grapes, soy milk", "label": 0},
        {"information": "vegan cream cheese, almond milk, tofu, eggs, cheese, milk, yogurt, vegetarian butter, honey, ghee, butter, fish", "label": 2},
        
        {"information": "A hearty chickpea and avocado salad with a citrus dressing is a refreshing dish. Toss chickpeas with diced avocado, cherry tomatoes, and red onion, then drizzle with a citrus dressing made from orange juice, olive oil, and mustard. This salad is rich in healthy fats and fiber.", "label": 0, "context": "Chickpeas, avocado, and citrus dressing are all plant-based, making this salad a nutritious option."},
        {"information": "Paneer Butter Masala. Ingredients: 200g paneer, 1 onion (chopped), 2 tomatoes (blended), 1/4 cup cream, 1 tbsp butter, 1 tsp ginger garlic paste, 1/2 tsp turmeric powder, 1/2 tsp garam masala, Salt to taste. Instructions: Sauté onions and ginger garlic paste in butter. Add tomatoes, turmeric powder, garam masala, and salt. Cook until oil separates. Add paneer cubes and cream. Simmer for 5-7 minutes. Serve with naan or rice.", "label": 1, "context": "This dish is vegetarian because it contains paneer (cottage cheese), cream, and butter, all dairy products derived from animals."},
        {"information": "Egg Curry. Ingredients: 4 boiled eggs, 1 onion (chopped), 2 tomatoes (blended), 1/4 cup yogurt, 1 tbsp oil, 1 tsp ginger garlic paste, 1/2 tsp cumin seeds, 1/2 tsp turmeric powder, 1/2 tsp chili powder, Salt to taste. Instructions: Sauté onions, ginger garlic paste, and cumin seeds in oil. Add tomatoes, turmeric, chili powder, and salt. Cook until oil separates. Add yogurt and boiled eggs. Simmer for 5 minutes. Serve with rice.", "label": 1, "context": "This curry is vegetarian as it includes eggs and yogurt, both animal-derived but not from killing animals."},
        {"information": "Chicken with Roasted Vegetables for Acne Patients: Ingredients: 1 chicken breast, 1 cup carrots, 1 cup zucchini, 1 tbsp olive oil, 1/4 tsp salt, 1/4 tsp pepper. Instructions: Roast vegetables with olive oil, salt, and pepper. Grill chicken breast. Serve together.", "label": 2, "context": "This meal is rich in protein and beta-carotene, which are essential for healthy skin and preventing acne."},
        {"information": "A roasted parsnip and carrot soup with thyme is a warming, comforting dish. Simmer parsnips and carrots with vegetable broth and thyme, then blend until smooth. This soup is rich in fiber and vitamins.", "label": 0, "context": "Parsnips and carrots are plant-based, making this soup a hearty vegan meal."},
        {"information": "A vegan sweet potato and black bean casserole is a comforting and satisfying dish. Layer sweet potatoes, black beans, onions, and spices in a casserole dish, then bake until golden. This casserole is full of fiber, protein, and essential vitamins.", "label": 0, "context": "Sweet potatoes and black beans are plant-based, making this casserole a delicious vegan meal."},
        {"information": "Vegetable Quiche. Ingredients: 1 pre-made pie crust, 3 eggs, 1 cup milk, 1/2 cup shredded cheddar cheese, 1/4 cup spinach, 1/4 cup bell pepper (diced), 1/4 cup onion (chopped), Salt and pepper to taste. Instructions: Preheat oven to 350°F. Whisk eggs, milk, cheese, and seasonings. Add vegetables and pour the mixture into the pie crust. Bake for 30-35 minutes. Serve warm.", "label": 1, "context": "This quiche is vegetarian as it contains eggs and cheese, which are both animal-based products."},
        {"information": "Grilled Chicken for Liver Disease Patients: Ingredients: 1 chicken breast, 1 tbsp olive oil, 1 tsp garlic, 1/2 tsp rosemary, 1/4 tsp black pepper. Instructions: Marinate chicken with olive oil, garlic, rosemary, and black pepper. Grill for 7-10 minutes.", "label": 2, "context": "Chicken is a lean source of protein, which is gentle on the liver and supports its healing."},
        {"information": "Grilled Salmon for Chronic Fatigue Syndrome Patients: Ingredients: 1 salmon fillet, 1 tbsp olive oil, 1/2 tsp lemon zest, 1/2 tsp black pepper. Instructions: Season salmon with olive oil, lemon zest, and black pepper. Grill for 7-10 minutes.", "label": 2, "context": "Salmon is rich in omega-3 fatty acids, which can help boost energy levels and combat fatigue."},

    
    ] 

    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)

        metrics = evaluate_model(model, tokenizer, test_data, batch_size=4)
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

