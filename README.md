
# SmartTag: Amazon ML Challenge 2025 - Smart Product Pricing

[](https://www.python.org/downloads/)
[](https://huggingface.co/transformers/)
[](https://pytorch.org/)

This repository contains the solution for the **Amazon ML Challenge 2025**, which focuses on predicting product prices from text descriptions. Our approach uses a fine-tuned `roberta-large` model to achieve a validation SMAPE score of **49.65%**.

## 📝 Project Overview

The core challenge is to predict a continuous numerical value (price) from unstructured and highly variable product catalog text. Our analysis revealed a wide distribution of prices and text lengths, indicating that a powerful language model was essential for accurate price estimation.

This solution leverages the deep linguistic understanding of a pre-trained transformer model while minimizing computational cost through a strategic fine-tuning approach called **gradual unfreezing**.

-----

## ⚙️ Methodology

### Problem Analysis

An initial analysis of the data revealed the following key characteristics:

  * **High Text Variability**: Product descriptions range from brief titles to detailed specifications, requiring a model robust to different text lengths and styles.
  * **Wide Price Range**: The significant variance in product prices suggests that subtle textual cues (brand names, materials, technical specs) are critical price determinants.
  * **Implicit Feature Engineering**: The model must implicitly learn features like product category, quality, and brand prestige directly from the text without explicit feature columns.

### Solution Strategy

We adopted a single-model strategy centered on transfer learning from a state-of-the-art Large Language Model (LLM).

  * **Approach Type**: Single Model (Fine-tuned Transformer)
  * **Core Innovation**: Our key innovation is the use of **gradual unfreezing**. Instead of fine-tuning the entire `roberta-large` model (355M+ parameters), we froze all layers except for the **last two transformer layers** and a custom regression head. This technique strikes a balance between adapting the model to the task and preserving the powerful, generalized representations learned during pre-training.
  * **Optimization**: The model was trained using Mean Squared Error (MSE) loss for its stability, while Symmetric Mean Absolute Percentage Error (SMAPE) was used as the primary metric for model selection and early stopping.

-----

## 🤖 Model Architecture

### Architecture Overview

The architecture is a sequential pipeline that processes raw text to predict a price. The `roberta-large` model acts as a sophisticated text feature extractor, and a multi-layer perceptron (MLP) head performs the final regression.

```
Input Text → RoBERTa Tokenizer → RoBERTa-large Base Model → [CLS] Token Embedding → Custom Regression Head (MLP) → Predicted Price
```

### Model Components

  * **Text Processing Pipeline**:
      * **Preprocessing**: Text is tokenized using the `roberta-large` tokenizer. Sequences are truncated or padded to a uniform maximum length of **386 tokens**.
      * **Model Type**: `roberta-large` from Hugging Face.
  * **Key Training Parameters**:
      * **Learning Rate**: $2e-5$
      * **Fine-tuning Strategy**: Only the last 2 transformer layers and the regression head are trainable.
      * **Loss Function**: Mean Squared Error (MSE)
      * **Optimizer**: AdamW
      * **Batch Size**: 64
  * **Image Processing Pipeline**: Not applicable. This solution is based solely on textual data.

-----

## 📊 Performance

The model was evaluated on a 20% validation split of the training data. The best performance was achieved based on the SMAPE score.

### Validation Results

| Metric | Score |
| :--- | :--- |
| **SMAPE Score** | **40.65%** |
| Mean Absolute Error (MAE) | $10.62 |
| R-squared ($R^2$) | $0.3616 |
| Pearson Correlation | $0.6061 |

These results demonstrate a strong correlation between the model's predictions and the true prices, indicating its effectiveness in capturing price-determining factors from the text.

-----

## 🚀 Conclusion

Our approach successfully demonstrates that a large, pre-trained transformer like `roberta-large` can be effectively and efficiently adapted for a specialized regression task. The **gradual unfreezing** technique proved crucial for achieving a strong result while managing computational resources, highlighting the power of transfer learning in transforming unstructured text into meaningful, predictive signals.
