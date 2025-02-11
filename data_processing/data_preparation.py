# This file handles the creation of clean dataset for MLM and NSP Training.
# It handles second level of parsing, where the extracted text from the first parsing is split into sentences.
# It ensures that each sentence is stored in a JSONL file and splits longer sentences into smaller chunks if they exceed the 512-token limit.

import os
import json
import re
import random
from logging import info as log

class DataPreparation:
    @staticmethod
    def split_sentence_into_chunks(sentence, max_tokens=512):
        """Splits a sentence into chunks with a maximum number of tokens (words)."""
        words = sentence.split()
        chunks, current_chunk, current_length = [], [], 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length <= max_tokens:
                current_chunk.append(word)
                current_length += word_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_length = [word], word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def process_all_files_in_directory(input_dir, output_dir, max_tokens=512):
        """Processes all files in a directory, extracting sentences and splitting long sentences."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for jsonl_file in os.listdir(input_dir):
            if jsonl_file.endswith(".jsonl"):
                input_path = os.path.join(input_dir, jsonl_file)
                output_path = os.path.join(output_dir, jsonl_file)

                with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                    for line in infile:
                        data = json.loads(line.strip())
                        sentence = data.get('sentence')
                        if not sentence:
                            continue 
                        chunks = DataPreparation.split_sentence_into_chunks(sentence, max_tokens)
                        for chunk in chunks:
                            outfile.write(json.dumps({'sentence': chunk}) + '\n')
                log(f"Processed file: {jsonl_file}")

    @staticmethod
    def combine_jsonl_files(input_dir, output_file):
        """Combines all individual JSONL files into one."""
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for jsonl_file in os.listdir(input_dir):
                if jsonl_file.endswith(".jsonl"):
                    input_path = os.path.join(input_dir, jsonl_file)
                    with open(input_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
                    outfile.write("\n")
                    log(f"Merged file: {jsonl_file}")
        log(f"MLM dataset: {output_file}")
        
    @staticmethod
    def clean_sentence(sentence):
        """Removes unwanted tags and trims whitespace from a sentence."""
        return sentence.strip('{\"sentence\":').strip('\"}').strip(' \"')
    
    @staticmethod
    def generate_nsp_pairs(input_file, output_file, start_index = 0):
        """Handles Next Sentence Prediction (NSP) pair generation."""
        sentences = []
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                sentence = data.get("sentence")
                if sentence:
                    sentences.append(sentence)
        if not sentences:
            return

        nsp_data = []
        for i in range(start_index, len(sentences) - 1):
            nsp_data.append({
                "sentence_a": sentences[i],
                "sentence_b": sentences[i + 1],
                "label": 1
            })
        for i in range(start_index, len(sentences) - 1):
            random_index = random.randint(0, len(sentences) - 1)
            while random_index == i or random_index == i + 1:
                random_index = random.randint(0, len(sentences) - 1)

            nsp_data.append({
                "sentence_a": sentences[i],
                "sentence_b": sentences[random_index],
                "label": 0
            })
        random.shuffle(nsp_data)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for record in nsp_data:
                record['sentence_a'] = DataPreparation.clean_sentence(record['sentence_a'])
                record['sentence_b'] = DataPreparation.clean_sentence(record['sentence_b'])              
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

        log(f"NSP dataset: {output_file}")
