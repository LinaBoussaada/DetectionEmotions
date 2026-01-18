
# 🎭 Multi-Label Emotion Detection in Text

Deep Learning system for detecting fine-grained emotions in text using the GoEmotions dataset.

## 📖 Overview
This project implements and compares four deep learning architectures for multi-label emotion classification on 58,000 Reddit comments annotated with 28 emotion categories.
Key Features:

Multi-label classification (texts can express multiple emotions)

4 architectures: LSTM, BiLSTM+Attention, CNN-BiLSTM-Attention, BERT

Comprehensive evaluation with 6 metrics

Explainability with LIME and attention visualization

Interactive web interface


## 🎯 Problem Statement
Unlike simple sentiment analysis (positive/negative), this system detects fine-grained emotions like joy, anger, surprise, gratitude, etc. The challenge lies in:

Multiple emotions per text (multi-label)
Severe class imbalance (92:1 ratio)
Subtle emotional expressions

## 📊 Results

<img width="698" height="261" alt="image" src="https://github.com/user-attachments/assets/c0d7ea2a-1cd5-42e1-8916-733c1a5cb723" />

### Key Findings
✅ BERT achieves best performance (61.5% F1-Micro)
✅ Hybrid CNN-BiLSTM offers best efficiency/performance trade-off
✅ Attention mechanism contributes +3.2% improvement
✅ Exact match remains challenging (< 60% even for BERT)

## 🧠 Architectures
1. LSTM Simple
Baseline model with unidirectional LSTM. Fast but limited context understanding.
2. BiLSTM with Attention
Bidirectional LSTM with attention mechanism. Captures context in both directions and focuses on important words.
3. CNN-BiLSTM-Attention (Hybrid)
Combines CNN for local feature extraction with BiLSTM for long-range dependencies. Best non-transformer model.
4. BERT (Transformer)
Fine-tuned BERT-base for multi-label classification. State-of-the-art performance with pre-trained knowledge.

## 📁 Dataset
GoEmotions by Google Research

58,009 Reddit comments
28 emotion categories (27 emotions + neutral)
Multi-label: 15.2% have multiple emotions
Languages: English

### Emotion Categories:

Positive (12): joy, gratitude, love, admiration, optimism, excitement, amusement, pride, approval, caring, desire, relief
Negative (11): anger, sadness, fear, disgust, disappointment, annoyance, disapproval, embarrassment, nervousness, grief, remorse
Ambiguous (4): surprise, confusion, curiosity, realization
Neutral (1): neutral

### 🔬 Ablation Study
Impact of each component (CNN-BiLSTM-Attention):
Component RemovedImpactBiLSTM-8.9% (Critical)CNN-5.0% (High)Attention-3.2% (Significant)Dropout-2.4% (Moderate)Pretrained Embeddings-0.8% (Low)
Conclusion: All components contribute, with BiLSTM being the most critical.

### 🔍 Explainability
LIME Analysis
Identifies which words influence predictions:

Example: "I'm so excited!" → [joy: 0.89, excitement: 0.76]
Important words: "excited" (+0.51), "so" (+0.23)

Attention Visualization
Heatmaps showing which tokens the model focuses on.


<div align="center">
⭐ If you found this project helpful, please star it!
Made with ❤️ for the NLP community
</div>
