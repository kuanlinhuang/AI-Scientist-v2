import warnings
from datetime import datetime
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    EsmTokenizer,
    EsmModel,
    EsmForProteinFolding,
    T5Tokenizer,
    T5EncoderModel,
    BertTokenizer,
    BertModel
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Define the path to the MAVE datasets
MAVE_DATA_DIR = os.path.join(os.path.dirname(__file__), "protein_language_models_data")

## DATASET REFERENCE

# If you want to use the UniRef50 dataset, you can refer to the following code:
uniref50 = load_dataset("uniref50", split="train")
# >>> uniref50.shape
# {'train': (45348397, 2)}

# If you want to use the TAPE benchmark datasets, you can refer to the following code:
tape_ss3 = load_dataset("tape", "secondary_structure", split="train")
# >>> tape_ss3.shape
# {'train': (8678, 3)}

tape_contact = load_dataset("tape", "contact", split="train")
# >>> tape_contact.shape
# {'train': (25299, 3)}

tape_stability = load_dataset("tape", "stability", split="train")
# >>> tape_stability.shape
# {'train': (53614, 3)}

# If you want to use the ProteinNet dataset, you can refer to the following code:
proteinnet = load_dataset("proteinnet", "casp12", split="train")
# >>> proteinnet.shape
# {'train': (34557, 4)}

# If you want to use the AlphaFold database, you can refer to the following code:
# Note: This is a large dataset and requires significant storage
# alphafold_db = load_dataset("alphafold_db", split="train")

## MAVE DATASETS REFERENCE

# The following MAVE (Multiplexed Assays of Variant Effects) datasets are available in the protein_language_models_data directory:
# 1. BRCA1_HUMAN_Findlay_2018.csv - BRCA1 variant effect data from Findlay et al. 2018
# 2. BRCA2_HUMAN_Erwood_2022_HEK293T.csv - BRCA2 variant effect data from Erwood et al. 2022
# 3. HXK4_HUMAN_Gersing_2022_activity.csv - Hexokinase 4 (Glucokinase) activity data from Gersing et al. 2022
# 4. CBX4_HUMAN_Tsuboyama_2023_2K28.csv - Chromobox protein homolog 4 data from Tsuboyama et al. 2023

def load_mave_dataset(dataset_name):
    """
    Load a MAVE dataset from the protein_language_models_data directory

    Args:
        dataset_name: Name of the dataset file (e.g., 'BRCA1_HUMAN_Findlay_2018.csv')

    Returns:
        DataFrame containing the MAVE dataset
    """
    file_path = os.path.join(MAVE_DATA_DIR, dataset_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {MAVE_DATA_DIR}")

    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {dataset_name} with {len(df)} variants")
    print(f"Columns: {df.columns.tolist()}")
    return df

# Example usage:
# brca1_data = load_mave_dataset('BRCA1_HUMAN_Findlay_2018.csv')
# brca2_data = load_mave_dataset('BRCA2_HUMAN_Erwood_2022_HEK293T.csv')
# hxk4_data = load_mave_dataset('HXK4_HUMAN_Gersing_2022_activity.csv')
# cbx4_data = load_mave_dataset('CBX4_HUMAN_Tsuboyama_2023_2K28.csv')

## PRE-TRAINED MODELS REFERENCE

## Example: load a pre-trained ESM-2 model
def load_esm2_model(model_name="facebook/esm2_t33_650M_UR50D"):
    """
    Load a pre-trained ESM-2 model and tokenizer

    Args:
        model_name: Name of the ESM-2 model to load

    Returns:
        tokenizer: ESM tokenizer
        model: ESM model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

## Example: load a pre-trained ESMFold model for protein structure prediction
def load_esmfold_model(model_name="facebook/esmfold_v1"):
    """
    Load a pre-trained ESMFold model for protein structure prediction

    Args:
        model_name: Name of the ESMFold model to load

    Returns:
        tokenizer: ESM tokenizer
        model: ESMFold model
    """
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name)
    return tokenizer, model

## Example: load a pre-trained ProtT5 model
def load_prot_t5_model(model_name="Rostlab/prot_t5_xl_uniref50"):
    """
    Load a pre-trained ProtT5 model and tokenizer

    Args:
        model_name: Name of the ProtT5 model to load

    Returns:
        tokenizer: T5 tokenizer
        model: T5 encoder model
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    return tokenizer, model

## Example: load a pre-trained ProtBERT model
def load_protbert_model(model_name="Rostlab/prot_bert"):
    """
    Load a pre-trained ProtBERT model and tokenizer

    Args:
        model_name: Name of the ProtBERT model to load

    Returns:
        tokenizer: BERT tokenizer
        model: BERT model
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model

## PROTEIN SEQUENCE DATASET CLASS

class ProteinSequenceDataset(Dataset):
    """
    Dataset class for protein sequences
    """
    def __init__(self, sequences, labels=None, tokenizer=None, max_length=1024):
        """
        Initialize the dataset

        Args:
            sequences: List of protein sequences
            labels: List of labels (optional)
            tokenizer: Tokenizer to use for encoding sequences
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        else:
            item = {"sequence": sequence}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

## EXAMPLE: FINE-TUNING A PROTEIN LANGUAGE MODEL FOR SECONDARY STRUCTURE PREDICTION

def fine_tune_plm_for_ss3(
    model_name="facebook/esm2_t12_35M_UR50D",
    batch_size=32,
    learning_rate=5e-5,
    num_epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Fine-tune a protein language model for secondary structure prediction (3-class)

    Args:
        model_name: Name of the pre-trained model to use
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs to train for
        device: Device to use for training

    Returns:
        model: Fine-tuned model
        metrics: Dictionary of training metrics
    """
    # Load the dataset
    dataset = load_dataset("tape", "secondary_structure")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Add a classification head for secondary structure prediction
    num_labels = 3  # 3 classes: helix, sheet, other
    model.classifier = nn.Linear(model.config.hidden_size, num_labels)
    model = model.to(device)

    # Create datasets
    train_data = ProteinSequenceDataset(
        train_dataset["primary"],
        train_dataset["ss3"],
        tokenizer
    )
    val_data = ProteinSequenceDataset(
        val_dataset["primary"],
        val_dataset["ss3"],
        tokenizer
    )
    test_data = ProteinSequenceDataset(
        test_dataset["primary"],
        test_dataset["ss3"],
        tokenizer
    )

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model.classifier(outputs.last_hidden_state[:, 0, :])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        metrics["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = model.classifier(outputs.last_hidden_state[:, 0, :])
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Accuracy: {val_accuracy:.4f}")

    return model, metrics

## EXAMPLE: PROTEIN EMBEDDING EXTRACTION

def extract_protein_embeddings(
    sequences,
    model_name="facebook/esm2_t33_650M_UR50D",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Extract embeddings from a protein language model

    Args:
        sequences: List of protein sequences
        model_name: Name of the pre-trained model to use
        batch_size: Batch size for inference
        device: Device to use for inference

    Returns:
        embeddings: Numpy array of protein embeddings
    """
    # Load the tokenizer and model
    tokenizer, model = load_esm2_model(model_name)
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = ProteinSequenceDataset(sequences, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding as sequence representation
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

## EXAMPLE USAGE FOR AI SCIENTIST

"""
# Example 1: Load a MAVE dataset and explore its contents
mave_dataset = load_mave_dataset('BRCA1_HUMAN_Findlay_2018.csv')
print(f"Dataset shape: {mave_dataset.shape}")
print(f"First few rows:\n{mave_dataset.head()}")

# Example 2: Extract embeddings for protein sequences in a MAVE dataset
mave_dataset = load_mave_dataset('BRCA1_HUMAN_Findlay_2018.csv')
sequences = mave_dataset['sequence'].tolist()[:100]  # Take first 100 sequences
embeddings = extract_protein_embeddings(sequences, model_name="facebook/esm2_t12_35M_UR50D")
print(f"Extracted embeddings shape: {embeddings.shape}")

# Example 3: Train a variant effect predictor using a MAVE dataset
model, metrics = train_variant_effect_predictor(
    dataset_name='HXK4_HUMAN_Gersing_2022_activity.csv',
    model_name="facebook/esm2_t12_35M_UR50D",
    batch_size=32,
    num_epochs=5
)

# Example 4: Compare performance of different protein language models on variant effect prediction
models_to_compare = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D"
]

results = {}
for model_name in models_to_compare:
    print(f"Training with model: {model_name}")
    model, metrics = train_variant_effect_predictor(
        dataset_name='BRCA1_HUMAN_Findlay_2018.csv',
        model_name=model_name,
        batch_size=32,
        num_epochs=3
    )
    results[model_name] = metrics['test_r2'][-1]

print("Final R² scores:")
for model_name, r2 in results.items():
    print(f"{model_name}: {r2:.4f}")
"""

## EXAMPLE: VARIANT EFFECT PREDICTION USING MAVE DATASETS

def train_variant_effect_predictor(
    dataset_name,
    model_name="facebook/esm2_t12_35M_UR50D",
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=10,
    test_size=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train a model to predict variant effects using a MAVE dataset

    Args:
        dataset_name: Name of the MAVE dataset file
        model_name: Name of the pre-trained protein language model to use
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs to train for
        test_size: Fraction of data to use for testing
        device: Device to use for training

    Returns:
        model: Trained model
        metrics: Dictionary of training metrics
    """
    # Load the MAVE dataset
    df = load_mave_dataset(dataset_name)

    # Extract sequences and labels (assuming the dataset has 'sequence' and 'score' columns)
    # Note: Column names may vary depending on the specific dataset
    if 'sequence' not in df.columns or 'score' not in df.columns:
        print(f"Warning: Expected 'sequence' and 'score' columns, but found: {df.columns.tolist()}")
        # Try to identify appropriate columns based on common patterns
        seq_cols = [col for col in df.columns if 'seq' in col.lower()]
        score_cols = [col for col in df.columns if any(term in col.lower() for term in ['score', 'effect', 'fitness', 'activity'])]

        if seq_cols and score_cols:
            print(f"Using '{seq_cols[0]}' for sequences and '{score_cols[0]}' for scores")
            sequences = df[seq_cols[0]].tolist()
            scores = df[score_cols[0]].values
        else:
            raise ValueError("Could not identify sequence and score columns in the dataset")
    else:
        sequences = df['sequence'].tolist()
        scores = df['score'].values

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, scores, test_size=test_size, random_state=42
    )

    # Load the tokenizer and model
    tokenizer, base_model = load_esm2_model(model_name)

    # Create a regression model by adding a regression head to the base model
    class ProteinRegressionModel(nn.Module):
        def __init__(self, base_model, hidden_size):
            super().__init__()
            self.base_model = base_model
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding as sequence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return self.regressor(cls_embedding).squeeze(-1)

    # Create the model
    model = ProteinRegressionModel(base_model, base_model.config.hidden_size)
    model = model.to(device)

    # Create datasets and dataloaders
    train_dataset = ProteinSequenceDataset(X_train, y_train, tokenizer)
    test_dataset = ProteinSequenceDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    metrics = {
        "train_loss": [],
        "test_loss": [],
        "test_r2": []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        metrics["train_loss"].append(train_loss)

        # Testing
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                predictions = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(predictions, labels)

                test_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        test_r2 = r2_score(all_labels, all_preds)
        metrics["test_loss"].append(test_loss)
        metrics["test_r2"].append(test_r2)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Test Loss: {test_loss:.4f} - "
              f"Test R²: {test_r2:.4f}")

    return model, metrics
