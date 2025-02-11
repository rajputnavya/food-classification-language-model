# import torch
# from datasets import load_dataset
# from transformers import TextDatasetForNextSentencePrediction
# from torch.utils.data import Dataset
# import logging
# import json

# class PreTrainingDataHandler:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def prepare_nsp_dataset(self, file_path, block_size=512):
#         """
#         Use TextDatasetForNextSentencePrediction for NSP data preparation.
#         """
#         try:
#             logging.info(f"Preparing NSP dataset from {file_path}")
#             return TextDatasetForNextSentencePrediction(
#                 tokenizer=self.tokenizer,
#                 file_path=file_path,
#                 block_size=block_size,
#                 overwrite_cache=True,
#             )
#         except Exception as e:
#             logging.error(f"Error preparing NSP dataset: {str(e)}")
#             raise

#     def prepare_mlm_dataset(self, file_path, max_length=512):
#         """
#         Prepare MLM dataset using Hugging Face's load_dataset utility.
#         """
#         try:
#             logging.info(f"Preparing MLM dataset from {file_path}")
#             mlm_data = load_dataset("json", data_files=file_path, split="train")
#             mlm_data = mlm_data.map(
#                 lambda x: self.tokenizer(
#                     x["sentence"],
#                     truncation=True,
#                     padding="max_length",
#                     max_length=max_length,
#                 ),
#                 batched=True,
#             )
#             mlm_data = mlm_data.map(lambda x: {"labels": x["input_ids"]}, batched=True)
#             mlm_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#             return mlm_data
#         except Exception as e:
#             logging.error(f"Error preparing MLM dataset: {str(e)}")
#             raise

#     def combine_datasets(self, mlm_dataset, nsp_dataset):
#         class CombinedDataset(Dataset):
#             def __init__(self, mlm_dataset, nsp_dataset):
#                 self.mlm_dataset = mlm_dataset
#                 self.nsp_dataset = nsp_dataset

#             def __len__(self):
#                 return max(len(self.mlm_dataset), len(self.nsp_dataset))

#             def __getitem__(self, idx):
#                 mlm_example = self.mlm_dataset[idx % len(self.mlm_dataset)]
#                 nsp_example = self.nsp_dataset[idx % len(self.nsp_dataset)]

#                 input_ids = mlm_example["input_ids"]
#                 attention_mask = mlm_example["attention_mask"]
#                 labels = mlm_example["labels"]

#                 token_type_ids = nsp_example.get("token_type_ids", torch.zeros_like(input_ids))
#                 token_type_ids = token_type_ids[:len(input_ids)]
#                 token_type_ids = torch.nn.functional.pad(token_type_ids, (0, len(input_ids) - len(token_type_ids)))

#                 next_sentence_label = nsp_example.get("next_sentence_label", -1)

#                 return {
#                     "input_ids": input_ids,
#                     "attention_mask": attention_mask,
#                     "token_type_ids": token_type_ids,
#                     "labels": labels,
#                     "next_sentence_label": next_sentence_label,
#                 }

#         logging.info("Combining MLM and NSP datasets")
#         return CombinedDataset(mlm_dataset, nsp_dataset)

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import logging

class PreTrainingDataHandler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_nsp_dataset(self, file_path, max_length=512):
        """
        Prepare NSP dataset using Hugging Face's `datasets` library.
        """
        try:
            logging.info(f"Preparing NSP dataset from {file_path}")
            nsp_data = load_dataset("json", data_files=file_path, split="train")

            def tokenize_nsp(example):
                """
                Tokenize the NSP data while retaining the `next_sentence_label`.
                Changes: Replaced deprecated `TextDatasetForNextSentencePrediction` with this approach.
                """
                # Modify to use 'sentence_a' and 'sentence_b' keys in NSP dataset
                encoding = self.tokenizer(
                    example["sentence_a"],  # Use 'sentence_a' for the first sentence
                    example["sentence_b"],  # Use 'sentence_b' for the second sentence
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
                # Retain `next_sentence_label` if provided in the dataset
                encoding["next_sentence_label"] = example.get("next_sentence_label", -1)  # Default to -1 if not provided
                return encoding

            # Apply tokenization to the dataset
            nsp_data = nsp_data.map(tokenize_nsp, batched=True)
            # Set the format for PyTorch compatibility
            nsp_data.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "next_sentence_label"])
            return nsp_data
        except Exception as e:
            logging.error(f"Error preparing NSP dataset: {str(e)}")
            raise

    def prepare_mlm_dataset(self, file_path, max_length=512):
        """
        Prepare MLM dataset using Hugging Face's `datasets` library.
        """
        try:
            logging.info(f"Preparing MLM dataset from {file_path}")
            mlm_data = load_dataset("json", data_files=file_path, split="train")

            def tokenize_mlm(example):
                """
                Tokenize MLM data and prepare `labels` for MLM training.
                Changes: Streamlined tokenization and ensured input IDs are duplicated as `labels`.
                """
                # Modify to use 'sentence' key in MLM dataset
                encoding = self.tokenizer(
                    example["sentence"],  # Use 'sentence' for the MLM task
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
                # Add `labels` for the MLM task, which are a copy of `input_ids`
                encoding["labels"] = encoding["input_ids"]
                return encoding

            # Apply tokenization to the dataset
            mlm_data = mlm_data.map(tokenize_mlm, batched=True)
            # Set the format for PyTorch compatibility
            mlm_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return mlm_data
        except Exception as e:
            logging.error(f"Error preparing MLM dataset: {str(e)}")
            raise

    def combine_datasets(self, mlm_dataset, nsp_dataset):
        """
        Combine MLM and NSP datasets into a single PyTorch dataset.
        """
        class CombinedDataset(Dataset):
            def __init__(self, mlm_dataset, nsp_dataset):
                """
                Initialize with MLM and NSP datasets.
                Changes: Adjusted `__getitem__` to handle mismatched lengths and provide padded `token_type_ids`.
                """
                self.mlm_dataset = mlm_dataset
                self.nsp_dataset = nsp_dataset

            def __len__(self):
                """
                Return the length of the combined dataset.
                Change: Using `max(len(mlm_dataset), len(nsp_dataset))` to ensure full coverage.
                """
                return max(len(self.mlm_dataset), len(self.nsp_dataset))

            def __getitem__(self, idx):
                """
                Fetch samples from MLM and NSP datasets.
                Changes: Added padding for `token_type_ids` and ensured consistency in sequence lengths.
                """
                # Use modulo to avoid IndexErrors in smaller dataset
                mlm_idx = idx % len(self.mlm_dataset)
                nsp_idx = idx % len(self.nsp_dataset)

                # Fetch examples from MLM and NSP datasets
                mlm_example = self.mlm_dataset[mlm_idx]
                nsp_example = self.nsp_dataset[nsp_idx]

                # Extract MLM-specific features
                input_ids = mlm_example["input_ids"]
                attention_mask = mlm_example["attention_mask"]
                labels = mlm_example["labels"]

                # Extract `token_type_ids` from NSP data or default to zeros
                token_type_ids = nsp_example.get("token_type_ids", torch.zeros_like(input_ids))
                # Ensure `token_type_ids` matches the length of `input_ids`
                token_type_ids = token_type_ids[:len(input_ids)]
                token_type_ids = torch.nn.functional.pad(token_type_ids, (0, len(input_ids) - len(token_type_ids)))

                # Extract `next_sentence_label` from NSP data
                next_sentence_label = nsp_example.get("next_sentence_label", -1)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": labels,
                    "next_sentence_label": next_sentence_label,
                }

        logging.info("Combining MLM and NSP datasets")
        return CombinedDataset(mlm_dataset, nsp_dataset)
