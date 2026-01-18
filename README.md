🎯 Overview
This project tackles multi-label emotion classification where texts can express multiple emotions simultaneously. We compare four deep learning architectures on the GoEmotions dataset (58,000 Reddit comments, 28 emotion categories).
Key Highlights

✅ 4 Architectures: LSTM, BiLSTM+Attention, Hybrid CNN-BiLSTM-Attention, BERT
✅ Comprehensive Evaluation: F1-Score, Precision, Recall, Hamming Loss, Exact Match
✅ Ablation Study: Quantifies impact of each architectural component
✅ Explainability: LIME integration + Attention visualization
✅ Interactive Demo: Gradio web interface for real-time predictions
✅ Publication Ready: Complete LaTeX report (10-15 pages)

✨ Features
Core Functionality

🔥 Multi-label Classification: Each text can have 0-N emotions
🧠 State-of-the-art Models: From simple LSTM to BERT
📊 Extensive Metrics: Micro/Macro F1, Hamming Loss, Exact Match
🎨 Visualizations: Learning curves, confusion matrices, attention heatmaps
🔍 Explainability: LIME explanations for model predictions
🌐 Web Interface: Interactive Gradio app with live predictions

Advanced Features

⚡ GPU Accelerated: CUDA support for fast training
💾 Model Persistence: Save/load trained models
📈 Training Monitoring: Real-time metrics tracking
🔧 Configurable: Easy hyperparameter tuning
📝 Well Documented: Comprehensive code comments

📊 Results
Model Comparison (Test Set)
ModelF1-MicroF1-MacroPrecisionRecallHamming LossExact MatchParamsLSTM Simple0.48230.31450.51240.45630.05230.41566.2MBiLSTM + Attn0.53410.38120.56780.50340.04710.47231.2MCNN-BiLSTM-Attn0.56230.41870.59120.53610.04340.50891.8MBERT0.61450.48340.64230.58910.03820.5712110M
Key Findings
✅ BERT achieves best performance (61.5% F1-Micro)
✅ Hybrid CNN-BiLSTM offers best efficiency/performance trade-off
✅ Attention mechanism contributes +3.2% improvement
✅ Exact match remains challenging (< 60% even for BERT)

🧠 Architectures
1. LSTM Simple
Baseline model with unidirectional LSTM. Fast but limited context understanding.
2. BiLSTM with Attention
Bidirectional LSTM with attention mechanism. Captures context in both directions and focuses on important words.
3. CNN-BiLSTM-Attention (Hybrid)
Combines CNN for local feature extraction with BiLSTM for long-range dependencies. Best non-transformer model.
4. BERT (Transformer)
Fine-tuned BERT-base for multi-label classification. State-of-the-art performance with pre-trained knowledge.

📁 Dataset
GoEmotions by Google Research

58,009 Reddit comments
28 emotion categories (27 emotions + neutral)
Multi-label: 15.2% have multiple emotions
Languages: English

Emotion Categories:

Positive (12): joy, gratitude, love, admiration, optimism, excitement, amusement, pride, approval, caring, desire, relief
Negative (11): anger, sadness, fear, disgust, disappointment, annoyance, disapproval, embarrassment, nervousness, grief, remorse
Ambiguous (4): surprise, confusion, curiosity, realization
Neutral (1): neutral

🎭 Multi-Label Emotion Detection in Text

Deep Learning system for detecting fine-grained emotions in text using the GoEmotions dataset.

Afficher l'image
Afficher l'image
Afficher l'image

📖 Overview
This project implements and compares four deep learning architectures for multi-label emotion classification on 58,000 Reddit comments annotated with 28 emotion categories.
Key Features:

Multi-label classification (texts can express multiple emotions)
4 architectures: LSTM, BiLSTM+Attention, CNN-BiLSTM-Attention, BERT
Comprehensive evaluation with 6 metrics
Explainability with LIME and attention visualization
Interactive web interface


🎯 Problem Statement
Unlike simple sentiment analysis (positive/negative), this system detects fine-grained emotions like joy, anger, surprise, gratitude, etc. The challenge lies in:

Multiple emotions per text (multi-label)
Severe class imbalance (92:1 ratio)
Subtle emotional expressions


📊 Results
Model Performance
ModelF1-MicroF1-MacroExact MatchParamsTraining TimeLSTM0.4820.3150.4166.2M~5 minBiLSTM+Attention0.5340.3810.4721.2M~6 minCNN-BiLSTM-Attention0.5620.4190.5091.8M~7 minBERT0.6150.4830.571110M~24 min
Key Findings
✅ BERT achieves best performance (61.5% F1-Micro)
✅ Hybrid CNN-BiLSTM offers best efficiency/performance trade-off
✅ Attention mechanism contributes +3.2% improvement
✅ Exact match remains challenging (< 60% even for BERT)

🧠 Architectures
1. LSTM Simple
Baseline model with unidirectional LSTM. Fast but limited context understanding.
2. BiLSTM with Attention
Bidirectional LSTM with attention mechanism. Captures context in both directions and focuses on important words.
3. CNN-BiLSTM-Attention (Hybrid)
Combines CNN for local feature extraction with BiLSTM for long-range dependencies. Best non-transformer model.
4. BERT (Transformer)
Fine-tuned BERT-base for multi-label classification. State-of-the-art performance with pre-trained knowledge.

📁 Dataset
GoEmotions by Google Research

58,009 Reddit comments
28 emotion categories (27 emotions + neutral)
Multi-label: 15.2% have multiple emotions
Languages: English

Emotion Categories:

Positive (12): joy, gratitude, love, admiration, optimism, excitement, amusement, pride, approval, caring, desire, relief
Negative (11): anger, sadness, fear, disgust, disappointment, annoyance, disapproval, embarrassment, nervousness, grief, remorse
Ambiguous (4): surprise, confusion, curiosity, realization
Neutral (1): neutral


🚀 Quick Start
Installation
bash# Clone repository
git clone https://github.com/votre-username/emotion-detection.git
cd emotion-detection

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py
Training
bash# Train BiLSTM (recommended for quick test)
python main.py --model bilstm_attention --epochs 5

# Train BERT (best performance)
python main.py --model bert --epochs 5 --batch_size 16
Demo
bash# Launch interactive interface
streamlit run app/streamlit_app.py
Notebook
Open notebooks/emotion_detection_complete.ipynb for complete walkthrough.

📈 Evaluation Metrics
We use multiple metrics tailored for multi-label classification:

F1-Score (Micro/Macro): Balance between precision and recall
Hamming Loss: Fraction of incorrectly predicted labels
Exact Match: Strictest metric - all labels must match perfectly
Precision/Recall: Per-class and overall performance
ROC-AUC: Discrimination capability

Why Exact Match matters: In multi-label, predicting [joy, excitement] when truth is [joy] counts as error.

🔬 Ablation Study
Impact of each component (CNN-BiLSTM-Attention):
Component RemovedImpactBiLSTM-8.9% (Critical)CNN-5.0% (High)Attention-3.2% (Significant)Dropout-2.4% (Moderate)Pretrained Embeddings-0.8% (Low)
Conclusion: All components contribute, with BiLSTM being the most critical.

🔍 Explainability
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
Component RemovedΔF1-MicroImpactAttention Mechanism-3.2%🔴 CriticalCNN Layers-5.0%🔴 HighBiLSTM-8.9%🔴 Very HighDropout-2.4%🟡 ModeratePretrained Embeddings-0.8%🟢 Low
Training Efficiency
ModelTime/Epoch (GPU)Total TrainingMemoryLSTM18s~5 min1.2 GBBiLSTM+Attn22s~6 min1.5 GBCNN-BiLSTM-Attn28s~7 min1.8 GBBERT95s~24 min4.2 GB
