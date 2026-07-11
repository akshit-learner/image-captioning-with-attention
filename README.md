# Image Captioning with Visual Attention

> An end-to-end deep learning project that generates natural language captions for images using an Encoder–Decoder architecture with Bahdanau Attention. The project includes data preprocessing, feature extraction, model training, evaluation, inference, and a Gradio-based web interface.

---

## Features

- CNN Encoder for visual feature extraction
- Bahdanau Attention mechanism
- GRU-based Decoder
- Image caption generation
- BLEU score evaluation
- Beam Search decoding
- Attention visualization during inference
- Interactive Gradio application
- Google Colab notebook for experimentation

---

## Model Architecture

The model follows an Encoder–Decoder architecture.

```text
Input Image
      │
      ▼
InceptionV3 Encoder
      │
Spatial Feature Maps
      │
      ▼
Bahdanau Attention
      │
Context Vector
      │
      ▼
GRU Decoder
      │
      ▼
Generated Caption
```

---

## Project Structure

```text
.
├── app/
│   └── app.py                           # Gradio application
│
├── notebooks/
│   └── Image_Captioning_Colab.ipynb     # Google Colab notebook
│
├── src/
│   ├── caption_model.py                 # CNN encoder, attention, and GRU decoder
│   ├── config.py                        # Configuration and project paths
│   ├── data_prep.py                     # Dataset preprocessing
│   ├── dataset.py                       # TensorFlow dataset pipeline
│   ├── evaluate.py                      # BLEU score evaluation
│   ├── features.py                      # Image feature extraction
│   ├── predict.py                       # Caption generation and inference
│   └── train.py                         # Model training pipeline
│
├── requirements.txt                     # Project dependencies
└── README.md
```

---

## Tech Stack

- Python
- TensorFlow
- NumPy
- Gradio
- InceptionV3
- Bahdanau Attention
- GRU
- BLEU Evaluation

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Project Workflow

### 1. Prepare the dataset

```bash
python data_prep.py
```

### 2. Extract image features

```bash
python features.py
```

### 3. Train the model

```bash
python train.py
```

### 4. Evaluate the model

```bash
python evaluate.py
```

### 5. Generate captions

```bash
python predict.py
```

### 6. Launch the web application

```bash
python app.py
```

---

## Repository Components

| File | Description |
|-------|-------------|
| `config.py` | Project configuration and paths |
| `data_prep.py` | Caption preprocessing and tokenizer creation |
| `features.py` | CNN feature extraction |
| `caption_model.py` | Encoder, Attention, and Decoder implementation |
| `dataset.py` | TensorFlow dataset pipeline |
| `train.py` | Model training |
| `evaluate.py` | BLEU score evaluation |
| `predict.py` | Caption generation |
| `app.py` | Gradio application |
| `Image_Captioning_Colab.ipynb` | Notebook implementation |

---

## Evaluation

The project includes an evaluation pipeline using BLEU scores for measuring caption quality.

---

## Inference

The inference pipeline supports:

- Image caption generation
- Beam Search decoding
- Attention visualization

---

## Notebook

A Google Colab notebook is included for running the complete workflow.

```text
notebooks/Image_Captioning_Colab.ipynb
```

---

## Requirements

All dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Train on larger image-caption datasets
- Improve caption quality with stronger vision encoders
- Experiment with Transformer-based decoders
- Add additional evaluation metrics

---

## License

This project is intended for educational and learning purposes.
