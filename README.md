# Chess Game with Machine Learning Evaluation

## 📋 Assignment Overview

This project implements a chess-playing application that **replaces the traditional Minimax algorithm's Piece-Square Table (PST) evaluation with a machine learning-based evaluation system**. The ML model uses a trained neural network to evaluate chess positions and select optimal moves.


## TESTING RESULTS 
<img width="799" height="580" alt="Testing w 500K data sample - 1" src="https://github.com/user-attachments/assets/cc15ef2c-9b11-44ab-838e-0f75ab003400" />

<img width="809" height="602" alt="Testing w 500K data sample - 2" src="https://github.com/user-attachments/assets/c81c3dd4-3f12-4c72-be9d-cf6120f24cbd" />

<img width="809" height="584" alt="Testing w 500K data sample" src="https://github.com/user-attachments/assets/aa9ea01a-e328-4cdb-8154-0fff924f82ae" />

---

## 🎯 Project Objectives

1. **Train a neural network** to evaluate chess positions using the Kaggle Chess Evaluations Dataset
2. **Replace the `evaluate_board_raw()` function** with ML-based evaluation
3. **Maintain the original Minimax structure** while enhancing position evaluation
4. **Provide graceful fallback** to traditional PST evaluation if ML model is unavailable

---

## 📁 Project Structure

```
HW2/
├── Chess_Minimax.ipynb          # Main notebook (modified)
├── archive/
│   └── chessData.csv            # Kaggle Chess Evaluations Dataset
├── chess_eval_model.pth         # Trained neural network weights (generated)
├── scaler.pkl                   # Score normalization scaler (generated)
└── README.md                    # This file
```

---

## 🚀 Setup Instructions

### Prerequisites

```bash
pip install chess ipywidgets pandas numpy scikit-learn torch tqdm
```

### Dataset

1. Download the **Chess Evaluations Dataset** from Kaggle:
   - URL: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
   - Extract to `archive/chessData.csv`

---

## 📊 Implementation Details

### Part 1: Data Preparation & Model Training

#### **Cell 4: Load Dataset**
- Loads chess positions from Kaggle dataset
- Uses 500K sample to prevent memory issues
- Dataset contains FEN positions with evaluation scores

#### **Cell 5: Feature Engineering**
- Converts FEN (Forsyth-Edwards Notation) to 64-element numerical vectors
- Maps pieces to values: P=1, N=3, B=3, R=5, Q=9, K=0 (negative for black)
- Empty squares = 0

#### **Cell 6: Data Cleaning**
- Removes special characters from evaluation scores (+, #, −)
- Converts to float values
- Drops invalid rows

#### **Cell 7: Normalization & Splitting**
- Normalizes evaluation scores to 0-1 range using MinMaxScaler
- Splits data: 90% training, 10% testing
- Saves scaler for inverse transformation

#### **Cell 8: Neural Network Architecture**
```
Input Layer:  64 features (board state)
Hidden Layer: 128 neurons + ReLU
Hidden Layer: 64 neurons + ReLU
Output Layer: 1 neuron (position evaluation)
```

#### **Cell 9: Model Training**
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Mean Squared Error (MSE)
- Epochs: 10
- Evaluates performance on test set

#### **Cell 10: Save Model**
- Saves trained model weights: `chess_eval_model.pth`
- Saves scaler: `scaler.pkl`

---

### Part 2: Chess Application with ML Evaluation

#### **Cell 2: Main Chess Game**

**Key Modification: `evaluate_board_raw()` Function**

**Original (PST-based):**
```python
def evaluate_board_raw(board: chess.Board) -> int:
    score = 0
    for sq, piece in board.piece_map().items():
        base = PIECE_VALUES[piece.piece_type]
        score += base + PST[piece.piece_type][sq]  # Uses Piece-Square Tables
    return score
```

**Modified (ML-based):**
```python
def evaluate_board_raw(board: chess.Board) -> int:
    """🤖 MODIFIED: Now uses ML-based evaluation"""
    if board.is_game_over():
        # Handle checkmate/draw
        return MATE_VALUE or -MATE_VALUE or 0
    
    if ML_AVAILABLE:
        # Use trained neural network
        with torch.no_grad():
            x = board_to_vec(board)
            normalized_score = ml_model(x).item()
            score = ml_scaler.inverse_transform([[normalized_score]])[0][0]
        return int(score)
    else:
        # Fallback to original PST evaluation
        # ... (original code)
```

**Features:**
- ✅ Uses neural network for position evaluation
- ✅ Handles game-over states (checkmate/draw)
- ✅ Inverse transforms normalized scores to original scale
- ✅ Graceful fallback to PST if ML model unavailable
- ✅ Error handling for robustness

---

## 🎮 How to Use

### Step 1: Train the ML Model (First Time Only)

Run cells in this order:
```
Cell 1  → Install dependencies
Cell 4  → Load dataset
Cell 5  → Convert FEN to features
Cell 6  → Clean data
Cell 7  → Normalize & split data
Cell 8  → Define neural network
Cell 9  → Train model
Cell 10 → Save model & scaler
```

**Expected Output:**
- Training progress with MSE loss per epoch
- Final test MSE score
- Saved files: `chess_eval_model.pth`, `scaler.pkl`

### Step 2: Play Chess

Run:
```
Cell 2 → Chess application with ML evaluation
```

**Game Controls:**
- **Mode:** Vs AI or Two Players
- **You:** Choose White or Black (vs AI mode)
- **AI depth:** 2-5 ply (higher = stronger but slower)
- **Start/Reset:** Begin new game
- **Undo:** Take back move(s)
- **Flip:** Rotate board view

### Step 3: Verify (Optional)

Run:
```
Cell 12 → Test evaluate_board_raw() function
```

Tests ML evaluation on:
- Starting position
- After 1. e4
- After 1. e4 e5
- Checkmate position (Fool's Mate)

---

## 🧠 Technical Details

### ML Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Feedforward Neural Network |
| Input Size | 64 (8×8 board) |
| Hidden Layers | 2 (128, 64 neurons) |
| Activation | ReLU |
| Output Size | 1 (evaluation score) |
| Loss Function | MSE |
| Optimizer | Adam (lr=0.001) |
| Training Data | 500K positions |
| Test Data | 50K positions |

### Evaluation Score Range

- **Original:** -15,312 to +15,319 (centipawns)
- **Normalized:** 0.0 to 1.0 (for training)
- **Output:** Inverse transformed to original scale

### Minimax Algorithm

- **Type:** Alpha-Beta Pruning
- **Depth:** 2-5 ply (configurable)
- **Move Ordering:** Captures, promotions, checks prioritized
- **Evaluation:** ML-based (with PST fallback)

---

## ✅ Assignment Requirements Met

- ✅ **Downloaded and explored** Kaggle chess evaluations dataset
- ✅ **Preprocessed data:**
  - Converted FEN positions to numerical features
  - Normalized evaluation scores
  - Split into training (90%) and test (10%) sets
- ✅ **Designed and trained neural network:**
  - Input: 64 features (board state)
  - Output: Position evaluation score
  - Architecture: 64 → 128 → 64 → 1
- ✅ **Evaluated model performance** using Mean Squared Error (MSE)
- ✅ **Saved trained model** for use in chess application
- ✅ **Replaced `evaluate_board_raw()` function** with ML-based evaluation
- ✅ **Maintained original notebook structure** (minimal modification)

---

## 🔧 Error Handling & Robustness

### Graceful Degradation

1. **PyTorch not installed** → Uses PST evaluation
2. **Model files not found** → Uses PST evaluation
3. **ML evaluation error** → Falls back to PST evaluation
4. **Memory issues** → Dataset sampling (500K positions)

### Safety Checks

```python
if ML_AVAILABLE and ml_model is not None and ml_scaler is not None and TORCH_AVAILABLE:
    # Use ML evaluation
else:
    # Use PST evaluation (fallback)
```

---

## 📈 Performance Comparison

### Traditional PST Evaluation
- **Speed:** Very fast (direct lookup)
- **Accuracy:** Good for tactical positions
- **Limitations:** Fixed heuristics, no learning

### ML-Based Evaluation
- **Speed:** Fast (single forward pass)
- **Accuracy:** Learned from 500K+ positions
- **Advantages:** Captures complex patterns, data-driven

---

## 🎯 Key Features

1. **Interactive UI** with ipywidgets
2. **Click-to-move** interface
3. **Move history log**
4. **Promotion dialog** for pawn promotions
5. **Board flip** for different perspectives
6. **Two game modes:** vs AI or two-player
7. **Adjustable AI strength** (depth 2-5)
8. **Undo functionality**
9. **Visual highlighting** for selected pieces and legal moves

---

## 📝 Code Modifications Summary

### Only Modified: `evaluate_board_raw()` Function

**Location:** Cell 2 (main chess application)

**Changes:**
1. Added ML model loading code
2. Added `board_to_vec()` helper function
3. Modified `evaluate_board_raw()` to use neural network
4. Added error handling and fallback logic
5. Kept all other code unchanged (UI, minimax, move ordering)

**Lines of Code:**
- Original function: ~15 lines
- Modified function: ~25 lines (with error handling)
- Additional setup: ~60 lines (model loading, helper functions)

---

## 🐛 Troubleshooting

### Issue: Kernel crashes when training

**Solution:** Dataset too large. Cell 4 now samples 500K positions.

### Issue: `evaluate_board_raw()` not found

**Solution:** Run Cell 2 before Cell 12 (verification).

### Issue: Model files not found

**Solution:** Run cells 4-10 to train and save the model first.

### Issue: Error when changing AI depth

**Solution:** Fixed by moving `board_to_vec()` inside `if TORCH_AVAILABLE:` block.

---

## 📚 References

- **Dataset:** [Chess Evaluations on Kaggle](https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations)
- **Chess Library:** [python-chess](https://python-chess.readthedocs.io/)
- **PyTorch:** [PyTorch Documentation](https://pytorch.org/docs/)
- **FEN Notation:** [Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)

---

## 👨‍💻 Author

**Course:** CS4200  
**Assignment:** HW2 - Chess Game with Machine Learning Evaluation  
**Date:** October 2025

---

## 📄 License

This project is for educational purposes as part of CS4200 coursework.

---

## 🎉 Conclusion

This project successfully demonstrates the integration of machine learning into a traditional game-playing algorithm. By replacing the static Piece-Square Table evaluation with a trained neural network, the chess engine can make more informed decisions based on patterns learned from hundreds of thousands of real chess positions.

**Key Achievement:** Minimal code modification (only `evaluate_board_raw()` function) while maintaining full backward compatibility with the original implementation.
