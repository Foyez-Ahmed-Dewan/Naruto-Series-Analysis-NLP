from transformers import pipeline
from nltk import sent_tokenize
import torch
import numpy as np
import pandas as pd

import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from utils import load_subtitles_dataset

import nltk
nltk.download('punkt')

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)
        
        
    def load_model(self, device):
        theme_classifier = pipeline("zero-shot-classification",
                                model = self.model_name,
                                device = device)
    
        return theme_classifier
    
    
    def get_themes_inference(self, script):
        # sentence tokenize
        script_sentences = sent_tokenize(script)

        # batch sentences
        sentence_batch_size = 20
        script_batches = []

        for index in range(0, len(script_sentences), sentence_batch_size):
        
            sent = " ".join(script_sentences[index: index + sentence_batch_size])
            script_batches.append(sent)

        # run model
        theme_output = self.theme_classifier(
            # script_batches[:2],
            script_batches,
            self.theme_list,
            multi_label=True
            )

        # wrangle output
        themes = {}

        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        themes= {key : np.mean(np.array(value)) for key, value in themes.items()}

        return themes
    

    def get_themes(self, dataset_path, save_path=None):
        #  Check if save_path exists
        if save_path is not None:
            if os.path.isdir(save_path):
                raise ValueError(f"Error: Expected a file but got a directory: {save_path}")

            if os.path.exists(save_path):
                print(f"Loading from existing file: {save_path}")
                df = pd.read_csv(save_path)
                return df

        # 🚀 Load dataset
        df = load_subtitles_dataset(dataset_path)

        # 🔍 Run inference
        output_themes = df['script'].apply(self.get_themes_inference)

        # Convert to DataFrame
        themes_df = pd.DataFrame(output_themes.tolist())
        df[themes_df.columns] = themes_df

        #  Save output if save_path is given
        if save_path is not None:
            if not save_path.endswith(".csv"):  # Ensure it's a valid file
                save_path = os.path.join(save_path, "output.csv")  
                print(f" Given a directory, saving as {save_path}")

            df.to_csv(save_path, index=False)
            print(f" Saved results to {save_path}")

        return df

