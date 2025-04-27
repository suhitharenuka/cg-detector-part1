# CpG Detector using LSTM

This project implements an LSTM-based deep learning model using PyTorch to predict the number of CpG sites (i.e., the number of 'CG' patterns) in randomly generated DNA sequences.

## Project Highlights
- **Data Generation**: Random DNA sequences are generated, and the true number of CpG sites is calculated.
- **Model Architecture**:
  - Embedding layer for nucleotide encoding.
  - LSTM layers to capture sequence patterns.
  - Fully connected layers for regression output.
- **Custom Loss Function**:
  - A weighted combination of Mean Squared Error (MSE) and L1 Loss to enhance model stability.
- **Training Setup**:
  - Optimizer: AdamW
  - Scheduler: ReduceLROnPlateau
  - Regularization techniques: Dropout and Gradient Clipping
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R² Score)

## How to Run

1. **Install Dependencies**:
```bash
pip install torch streamlit pyngrok
```

2. **Train the Model**:
```bash
python app.py
```


