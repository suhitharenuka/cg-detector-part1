import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Tuple, Sequence
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import random

# DO NOT CHANGE HERE
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

# Use this for getting x label
def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:
    for i in range(n_seqs):
        yield [random.randint(0, 4) for _ in range(seq_len)]

# Use this for getting y label
def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        # note that seq is a string, not a list
        if dimer == "CG":
            cgs += 1
    return cgs

# Alphabet helpers
alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(5))}
int2dna = { i: a for a, i in zip(alphabet, range(5))}

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# we prepared two datasets for training and evaluation
# training data scale we set to 2048
# we test on 512

def prepare_data(num_samples=100):
    # prepared the training and test data
    # you need to call rand_sequence and count_cpgs here to create the dataset
    # step 1
    X_dna_seqs_train = list(rand_sequence(num_samples))
    #step2
    temp = ["".join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs_train] # use intseq_to_dnaseq here to convert ids back to DNA seqs
    #step3
    y_dna_seqs = [count_cpgs(seq) for seq in temp] # use count_cpgs here to generate labels with temp generated in step2
    return torch.tensor(X_dna_seqs_train, dtype=torch.long), torch.tensor(y_dna_seqs, dtype=torch.float)

    #return X_dna_seqs_train, y_dna_seqs

train_x, train_y = prepare_data(2048)
test_x, test_y = prepare_data(512)

# chosen config
LSTM_HIDDEN = 64
LSTM_LAYER = 3
batch_size = 32
learning_rate = 0.003
epoch_num = 30

# create data loader

train_data_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)


class CpGPredictor(nn.Module):
    """LSTM-based model for predicting CpG counts in DNA sequences"""
    def __init__(self,
                 input_size=5,      # Size of vocabulary (N,A,C,G,T = 5)
                 embedding_dim=64,   # Increased embedding dimension
                 hidden_size=128,    # Increased hidden size
                 num_layers=2,
                 dropout=0.3):       # Increased dropout for better regularization
        super(CpGPredictor, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim)

        # Bidirectional LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Enhanced prediction head with more layers and batch normalization
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # Input shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate final hidden states from both directions
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Prediction head
        output = self.predictor(final_hidden)

        # Apply ReLU to ensure non-negative counts
        output = F.relu(output)

        return output.squeeze()

# init model / loss function / optimizer etc.
# Custom loss function combining MSE and L1 loss
class CpGLoss(nn.Module):
    def __init__(self, mse_weight=0.7, l1_weight=0.3):
        super(CpGLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        return self.mse_weight * mse_loss + self.l1_weight * l1_loss

def evaluate_model(model: nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[float, float, float, float]:
    """
    Evaluate model performance using multiple metrics

    Returns:
        Tuple of (MAE, MSE, RMSE, R2 Score)
    """
    model.eval()
    true_counts = []
    pred_counts = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            # Round predictions to nearest integer
            rounded_preds = outputs.round()

            true_counts.extend(y_batch.cpu().numpy())
            pred_counts.extend(rounded_preds.cpu().numpy())

    # Convert to numpy arrays
    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)

    # Calculate metrics
    mae = mean_absolute_error(true_counts, pred_counts)
    mse = mean_squared_error(true_counts, pred_counts)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_counts, pred_counts)

    return mae, mse, rmse, r2

def sequence_to_tensor(sequence: str) -> torch.Tensor:
    """Convert DNA sequence string to tensor"""
    # Mapping for nucleotides to indices
    nuc_to_idx = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

    # Convert sequence to indices
    indices = [nuc_to_idx[nuc] for nuc in sequence.upper()]

    # Convert to tensor
    return torch.tensor(indices).unsqueeze(0)  # Add batch dimension

def count_cpg(sequence: str) -> int:
    """Count actual CpG occurrences in sequence"""
    return sequence.upper().count('CG')

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CpGPredictor()
model.load_state_dict(torch.load('best_cpg_predictor.pt', map_location=device))
model.to(device)
model.eval()

# Streamlit App
st.title("CpG Prediction in DNA Sequences")
st.markdown("""
This app predicts the number of CpG counts in a given DNA sequence.
Enter a DNA sequence and compare the actual count with the model's prediction.
""")

# User Input
sequence = st.text_input("Enter DNA Sequence:", "")

if sequence:
    if all(nuc in "NACGT" for nuc in sequence.upper()):
        seq_tensor = sequence_to_tensor(sequence).to(device)
        with torch.no_grad():
            pred = model(seq_tensor)
            predicted_count = pred.round().item()
        actual_count = count_cpg(sequence)
        difference = abs(predicted_count - actual_count)

        # Display Results
        st.subheader("Results")
        st.write(f"**Entered Sequence:** {sequence}")
        st.write(f"**Predicted CpG Count:** {predicted_count}")
        st.write(f"**Actual CpG Count:** {actual_count}")
        st.write(f"**Difference:** {difference}")
    else:
        st.error("Invalid DNA sequence. Only characters N, A, C, G, T are allowed.")

