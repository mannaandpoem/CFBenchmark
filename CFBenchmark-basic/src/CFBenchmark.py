import os
import pandas as pd
import numpy as np
import torch
import argparse
import pickle
from openai import OpenAI
import time
from typing import Dict, List, Any, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading
from modelscope import AutoModel, AutoTokenizer

class CFBenchmark:
    def __init__(self,
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 model_name: str = "gpt-3.5-turbo",
                 fewshot_text_path: str = "../fewshot",
                 test_type: str = "few-shot",
                 response_path: str = "../cfbenchmark-response",
                 scores_path: str = "../cfbenchmark-scores",
                 benchmark_path: str = "../data",
                 embedding_model_path: str = "BAAI/bge-large-zh-v1.5",
                 max_workers: int = 4,
                 api_rate_limit: float = 1.0
                 ) -> None:
        
        # Set up OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.modelname = model_name
        self.fewshot_text_path = fewshot_text_path
        self.test_type = test_type
        self.response_path = response_path
        self.scores_path = scores_path
        self.benchmark_path = benchmark_path
        self.embedding_model_path = embedding_model_path
        self.max_workers = max_workers
        self.api_rate_limit = api_rate_limit  # seconds between API calls
        self._rate_limit_lock = threading.Lock()
        self._last_api_call = 0

        self.classifications = [
            'company', 'product', 'sector', 'event',
            'sentiment', 'summary', 'risk', 'suggestion'
        ]

        self.fewshot_text = {}
        if test_type == 'few-shot':
            for item in self.classifications:
                filename = 'fewshot-' + item + '.txt'
                try:
                    with open(os.path.join(fewshot_text_path, filename), 'r', encoding='utf-8') as file:
                        content = file.read()
                        self.fewshot_text[item] = content
                except FileNotFoundError:
                    print(f"Warning: Few-shot file not found: {os.path.join(fewshot_text_path, filename)}")
                    self.fewshot_text[item] = ""

        # Load labels
        try:
            labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labels_info.pkl")
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as file:
                    self.labels = pickle.load(file)
            else:
                # Try relative path
                labels_path = os.path.join(os.path.dirname(self.benchmark_path), "labels_info.pkl")
                with open(labels_path, 'rb') as file:
                    self.labels = pickle.load(file)
        except FileNotFoundError:
            print(f"Warning: labels_info.pkl not found, using empty labels dictionary")
            self.labels = {cls: [] for cls in self.classifications}
        
        # Load embedding model using modelscope (following the provided example pattern)
        self.t2v_tokenizer = None
        self.t2v_model = None
        try:
            print(f"Loading embedding model from: {self.embedding_model_path}")
            self.t2v_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
            self.t2v_model = AutoModel.from_pretrained(self.embedding_model_path)
            self.t2v_model.eval()
            print("✓ Loaded embedding model using modelscope")
        except Exception as e:
            print(f"⚠️  Warning: Error loading embedding model ({str(e)}). Using enhanced TF-IDF based cosine similarity instead.")
            print("   To use the embedding model, ensure the model path is correct and modelscope is installed.")
            self.t2v_tokenizer = None
            self.t2v_model = None
    
    @staticmethod
    def install_optional_dependencies():
        """Helper method to install optional dependencies for better performance."""
        try:
            import subprocess
            import sys
            
            print("Installing optional dependencies for enhanced performance...")
            
            # Install modelscope
            subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
            
            # Install torch if not available
            try:
                import torch
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
            
            # Install datasets if not available
            try:
                import datasets
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            
            print("✓ Optional dependencies installed successfully!")
            print("  Please restart your Python session to use the enhanced features.")
            
        except Exception as e:
            print(f"❌ Error installing dependencies: {str(e)}")
            print("   Please install manually: pip install modelscope torch datasets")
    
    def _rate_limited_api_call(self, instruction: str, classes: str) -> str:
        """Rate-limited API call wrapper for thread safety."""
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_api_call
            if time_since_last_call < self.api_rate_limit:
                time.sleep(self.api_rate_limit - time_since_last_call)
            self._last_api_call = time.time()
        
        # Use OpenAI API for generation
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.modelname,
                    messages=[
                        {"role": "user", "content": instruction}
                    ],
                    max_tokens=512 if (classes == 'summary' or classes == 'suggestion' or classes == 'risk') else 64,
                    temperature=0.0
                )
                
                generated_text = response.choices[0].message.content.strip()
                return generated_text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return "API Error"
        
        return "API Error"
    
    def get_row_response(self, row: Dict[str, Any], classes: str, types: str) -> str:
        context = row['input']    
        instruction = ''
        
        if types == 'zero-shot':
            instruction = row['instruction'] + context
        else:
            instruction = self.fewshot_text[classes]
            case = '\ncase4：\n新闻内容：' + context
            if classes == 'sector' or classes == 'event' or classes == 'sentiment':
                try:
                    labels = row['instruction'].split('（', 1)[1]
                    labels = labels.split('）', 1)[0]
                    case = case + '\n类别：（' + labels + '）\n'
                except (IndexError, KeyError) as e:
                    print(f"Warning: Could not parse instruction: {row.get('instruction', 'None')}")
            instruction = instruction + case

        instruction = instruction + '\n回答：'
        
        # Use rate-limited API call
        generated_text = self._rate_limited_api_call(instruction, classes)
        
        # Process response to match the expected format
        if types == 'zero-shot':
            if '回答：' in generated_text:
                generated_text = generated_text.split('回答：', 1)[-1]
        else:
            if '回答：' in generated_text:
                generated_text = generated_text.split('回答：', 4)[-1]
        
        generated_text = generated_text.split('\n', 1)[0].strip()
        return generated_text

    def load_benchmark_data(self, item):
        """Load benchmark data from files directly instead of using datasets library."""
        try:
            # Create sample data structure in case we need it
            sample_data = [{
                "input": "这是一个示例输入", 
                "instruction": "这是一个示例指令", 
                "response": "示例响应"
            }]
            
            # Try different file locations and formats
            possible_paths = [
                os.path.join(self.benchmark_path, f"{item}.json"),
                os.path.join(self.benchmark_path, f"{item}.csv"),
                os.path.join(self.benchmark_path, item, "data.json"),
                os.path.join(self.benchmark_path, item, "data.csv"),
            ]
            
            # Add parent directory options
            parent_dir = os.path.dirname(self.benchmark_path)
            possible_paths.extend([
                os.path.join(parent_dir, f"{item}.json"),
                os.path.join(parent_dir, f"{item}.csv"),
                os.path.join(parent_dir, "data", f"{item}.json"),
                os.path.join(parent_dir, "data", f"{item}.csv"),
            ])
            
            # Try each path
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Found data file: {path}")
                    if path.endswith('.json'):
                        with open(path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                                # Handle different JSON structures
                                if isinstance(data, list):
                                    return pd.DataFrame(data)
                                elif isinstance(data, dict):
                                    if "data" in data:
                                        return pd.DataFrame(data["data"])
                                    # Try to convert the dictionary itself
                                    try:
                                        return pd.DataFrame([data])
                                    except:
                                        pass
                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON from {path}")
                    elif path.endswith('.csv'):
                        try:
                            return pd.read_csv(path)
                        except Exception as e:
                            print(f"Failed to load CSV from {path}: {str(e)}")
            
            # If we reach here, create an example dataset for testing
            print(f"No data files found for {item}. Creating sample data for testing.")
            return pd.DataFrame(sample_data)
            
        except Exception as e:
            print(f"Error in load_benchmark_data for {item}: {str(e)}")
            # Return sample data as fallback
            return pd.DataFrame(sample_data)

    def _process_single_row(self, row_data: tuple) -> tuple:
        """Helper method to process a single row for parallel execution."""
        index, row, item, test_type = row_data
        try:
            output = self.get_row_response(row, item, test_type)
            return index, output
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            return index, "API Error"

    def get_model_results(self):
        save_dir = os.path.join(self.response_path, self.test_type)
        save_dir = os.path.join(save_dir, self.modelname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"benchmark_path: {self.benchmark_path}")
        # self.classifications = ["suggestion"]
        print(f"classifications: {self.classifications}")
        
        for item in self.classifications:
            print(f'Processing {item}')
            
            # Try to load data with better error handling
            df = None
            
            # Check if this is a HuggingFace datasets structure
            try:
                from datasets import load_from_disk
                
                # First check if the main path is a dataset directory
                main_dataset_info = os.path.join(self.benchmark_path, "dataset_info.json")
                if os.path.exists(main_dataset_info):
                    try:
                        dataset = load_from_disk(self.benchmark_path)
                        if item in dataset:
                            df = dataset[item].to_pandas()
                            print(f"Successfully loaded data using datasets library from main directory")
                        else:
                            print(f"Item '{item}' not found in main dataset")
                    except Exception as e:
                        print(f"Error loading from main dataset: {str(e)}")
                
                # If that didn't work, try loading individual subdirectories
                if df is None:
                    item_path = os.path.join(self.benchmark_path, item)
                    item_dataset_info = os.path.join(item_path, "dataset_info.json")
                    if os.path.exists(item_dataset_info):
                        try:
                            dataset = load_from_disk(item_path)
                            df = dataset.to_pandas()
                            print(f"Successfully loaded data using datasets library from {item} subdirectory")
                        except Exception as e:
                            print(f"Error loading from {item} subdirectory: {str(e)}")
                
                # If still no data, fall back to direct file loading
                if df is None:
                    print("Falling back to direct file loading")
                    df = self.load_benchmark_data(item)
                    
            except ImportError:
                print("datasets library not available, using direct file loading")
                df = self.load_benchmark_data(item)
            except Exception as e:
                print(f"Unexpected error with datasets: {str(e)}")
                df = self.load_benchmark_data(item)
            
            # # FIXME: Test: first 50 for df
            # df = df.head(50)

            # Ensure we have a valid dataframe with required columns
            if df is None or df.empty:
                print(f"Creating sample data for {item}")
                df = pd.DataFrame([
                    {
                        "input": "这是一个示例输入", 
                        "instruction": "这是一个示例指令", 
                        "response": "示例响应"
                    }
                ])
            
            # Make sure required columns exist
            for col in ['input', 'instruction', 'response']:
                if col not in df.columns:
                    df[col] = "" if col != 'input' else "这是一个示例输入"
            
            print(f"Processing {len(df)} examples for {item}")
            
            # Process each row with parallel execution
            outputs = [""] * len(df)  # Pre-allocate output list
            
            # Prepare data for parallel processing
            row_data_list = [(i, row, item, self.test_type) for i, row in df.iterrows()]
            
            # Use ThreadPoolExecutor for parallel API calls
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(self._process_single_row, row_data): row_data[0] 
                    for row_data in row_data_list
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_index):
                    try:
                        index, output = future.result()
                        outputs[index] = output
                        completed_count += 1
                        
                        if completed_count % 5 == 0:
                            print(f"Processed {completed_count}/{len(df)} examples for {item}")
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"Error processing row {index}: {str(e)}")
                        outputs[index] = "API Error"
            
            df['output'] = outputs
            
            # Save results
            result_df = df[['input', 'response', 'output']]
            filename = item + '-output.csv'
            savepath = os.path.join(save_dir, filename)
            result_df.to_csv(savepath, index=False)
            print(f'Saved results for {item} to {savepath}')

    def get_y(self, row, label_list):
        y_true = np.zeros((len(label_list) + 1, 1))
        y_pred = np.zeros((len(label_list) + 1, 1))
        response = set([item.strip() for item in str(row['response']).replace('，', ',').strip().split(',') if item])
        output = set([item.strip() for item in str(row['output']).replace('，', ',').strip().split(',') if item])   

        for i in range(len(label_list)):
            if label_list[i] in response:
                y_true[i] = 1
            if label_list[i] in output:
                y_pred[i] = 1
        
        if y_pred.sum() == 0 or len(output) > y_pred.sum():
            y_pred[-1] = 1
        return y_true, y_pred

    def get_f1_score(self, row, label_list):
        y_true, y_pred = self.get_y(row, label_list=label_list)
        prec = (y_true * y_pred).sum() / max(y_true.sum(), 1e-10)
        reca = (y_true * y_pred).sum() / max(y_pred.sum(), 1e-10)
        if prec == 0 or reca == 0:
            f1 = 0
        else:
            f1 = 2 * prec * reca / (prec + reca)
        return f1

    def _process_cosine_similarity_batch(self, rows_batch: List[tuple]) -> List[tuple]:
        """Process a batch of rows for cosine similarity calculation."""
        results = []
        for index, row in rows_batch:
            try:
                cosine_s = self.get_cosine_similarities(row)
                results.append((index, cosine_s))
            except Exception as e:
                print(f"Error calculating cosine similarity for row {index}: {str(e)}")
                results.append((index, 0.0))
        return results

    def _process_f1_score_batch(self, rows_batch: List[tuple], label_list: List[str]) -> List[tuple]:
        """Process a batch of rows for F1 score calculation."""
        results = []
        for index, row in rows_batch:
            try:
                f1_score = self.get_f1_score(row, label_list)
                results.append((index, f1_score))
            except Exception as e:
                print(f"Error calculating F1 score for row {index}: {str(e)}")
                results.append((index, 0.0))
        return results

    def get_cosine_similarities(self, row):
        sentences_1 = str(row['output'])
        sentences_2 = str(row['response'])
        
        # Use modelscope embedding model if available (following the provided example pattern)
        if self.t2v_model is not None and self.t2v_tokenizer is not None:
            try:
                sentences = [sentences_1, sentences_2]
                
                # Tokenize sentences
                encoded_input = self.t2v_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.t2v_model(**encoded_input)
                    # Perform pooling. In this case, cls pooling.
                    sentence_embeddings = model_output[0][:, 0]
                
                # normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                
                # Calculate cosine similarity between the two embeddings
                cosine_sim = torch.nn.functional.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
                return cosine_sim.item()
            except Exception as e:
                print(f"Error with modelscope embedding model: {str(e)}")
                # Fall back to simple method
        
        # Enhanced fallback method using TF-IDF based cosine similarity
        try:
            # Text preprocessing
            def preprocess_text(text):
                # Remove punctuation and normalize
                text = text.lower()
                # Replace Chinese punctuation
                text = text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ')
                text = text.replace('、', ' ').replace('；', ' ').replace('：', ' ').replace('"', ' ')
                text = text.replace('"', ' ').replace(''', ' ').replace(''', ' ').replace('（', ' ').replace('）', ' ')
                # Replace English punctuation
                text = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
                text = text.replace(';', ' ').replace(':', ' ').replace('"', ' ').replace("'", ' ')
                text = text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
                # Remove extra spaces
                text = ' '.join(text.split())
                return text
            
            text1 = preprocess_text(sentences_1)
            text2 = preprocess_text(sentences_2)
            
            # Handle empty texts
            if not text1.strip() and not text2.strip():
                return 1.0  # Both empty, perfect match
            if not text1.strip() or not text2.strip():
                return 0.0  # One empty, no match
            
            # Split into tokens for Chinese and English
            def get_tokens(text):
                # For mixed Chinese-English text, use both word and character level
                words = text.split()
                tokens = []
                for word in words:
                    # If word contains Chinese characters, split into characters
                    if any('\u4e00' <= char <= '\u9fff' for char in word):
                        tokens.extend(list(word))
                    else:
                        tokens.append(word)
                return tokens
            
            tokens1 = get_tokens(text1)
            tokens2 = get_tokens(text2)
            
            # Create vocabulary
            vocab = list(set(tokens1 + tokens2))
            if not vocab:
                return 0.0
            
            # Calculate simple TF vectors (term frequency)
            def calculate_tf_vector(tokens, vocab):
                tf = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                
                # Create vector based on vocabulary
                vector = []
                for token in vocab:
                    vector.append(tf.get(token, 0))
                
                return np.array(vector, dtype=float)
            
            vector1 = calculate_tf_vector(tokens1, vocab)
            vector2 = calculate_tf_vector(tokens2, vocab)
            
            # Calculate cosine similarity
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Ensure the result is between 0 and 1
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            
            return cosine_sim
            
        except Exception as e:
            print(f"Error in enhanced similarity calculation: {str(e)}")
            # Final fallback to simple Jaccard similarity
            try:
                # Simple word-level Jaccard similarity as last resort
                words1 = set(sentences_1.lower().replace('，', ' ').replace('。', ' ').split())
                words2 = set(sentences_2.lower().replace('，', ' ').replace('。', ' ').split())
                
                if not words1 and not words2:
                    return 1.0
                if not words1 or not words2:
                    return 0.0
                
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                
                return intersection / union if union > 0 else 0.0
                
            except Exception as e2:
                print(f"Error in fallback Jaccard similarity: {str(e2)}")
                return 0.0

    def get_test_scores(self):
        result_directory = os.path.join(self.scores_path, self.test_type, self.modelname)
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
            
        for classes in self.classifications:
            filename = classes + '-output.csv'
            response_file_path = os.path.join(self.response_path, self.test_type, self.modelname, filename)
            
            if not os.path.exists(response_file_path):
                print(f"Warning: Response file not found for {classes}, skipping evaluation")
                continue
                
            df = pd.read_csv(response_file_path)
            
            if df.empty:
                print(f"Warning: Empty response file for {classes}, skipping evaluation")
                continue
                
            if classes == 'suggestion' or classes == 'summary' or classes == 'risk':
                # Parallel processing for cosine similarity
                batch_size = max(1, len(df) // self.max_workers)
                batches = []
                for i in range(0, len(df), batch_size):
                    batch = [(idx, row) for idx, row in df.iloc[i:i+batch_size].iterrows()]
                    batches.append(batch)
                
                cosine_scores = [0.0] * len(df)
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self._process_cosine_similarity_batch, batch): batch 
                        for batch in batches
                    }
                    
                    for future in as_completed(future_to_batch):
                        try:
                            batch_results = future.result()
                            for index, score in batch_results:
                                cosine_scores[index] = score
                        except Exception as e:
                            print(f"Error processing cosine similarity batch: {str(e)}")
                
                df['cosine_s'] = cosine_scores
                score1 = df['cosine_s'].sum() / len(df)
                result_message = f"{self.modelname}的{classes} cosine_similarity为{score1}"
                print(result_message)
                
            elif classes == 'company' or classes == 'product':
                # For company/product, use response labels directly
                df['f1score'] = df.apply(lambda row: self.get_f1_score(row, row['response'].split('，')), axis=1)
                score1 = df['f1score'].sum() / len(df)
                print(f"{self.modelname}的{classes} f1 score 为{score1}")
                
            else:
                # Dynamically extract labels from the data instead of using predefined labels
                all_labels = set()
                
                # Extract labels from responses and outputs
                for _, row in df.iterrows():
                    # Handle response labels
                    if pd.notna(row['response']):
                        response_labels = [label.strip() for label in str(row['response']).replace('，', ',').split(',') if label.strip()]
                        all_labels.update(response_labels)
                    
                    # Handle output labels  
                    if pd.notna(row['output']):
                        output_labels = [label.strip() for label in str(row['output']).replace('，', ',').split(',') if label.strip()]
                        all_labels.update(output_labels)
                
                # Convert to sorted list for consistent ordering
                label_list = sorted(list(all_labels))
                print(f"Found {len(label_list)} unique labels for {classes}: {label_list}")
                
                if label_list:
                    # Parallel processing for F1 scores
                    batch_size = max(1, len(df) // self.max_workers)
                    batches = []
                    for i in range(0, len(df), batch_size):
                        batch = [(idx, row) for idx, row in df.iloc[i:i+batch_size].iterrows()]
                        batches.append(batch)
                    
                    f1_scores = [0.0] * len(df)
                    
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        future_to_batch = {
                            executor.submit(self._process_f1_score_batch, batch, label_list): batch 
                            for batch in batches
                        }
                        
                        for future in as_completed(future_to_batch):
                            try:
                                batch_results = future.result()
                                for index, score in batch_results:
                                    f1_scores[index] = score
                            except Exception as e:
                                print(f"Error processing F1 score batch: {str(e)}")
                    
                    df['f1score'] = f1_scores
                    score1 = df['f1score'].sum() / len(df)
                    print(f"{self.modelname}的{classes} f1 score 为{score1}")
                else:
                    print(f"Warning: No labels found for {classes}, skipping evaluation")
                    continue

            filename = classes + '-scores.csv'
            df.to_csv(os.path.join(result_directory, filename))
