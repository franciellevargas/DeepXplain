# Project Timeline: DeepXplain Summer 2025
---
**Students:**  
- Røskva Bjørgfinsdóttir — [rosktb@gmail.com](mailto:rosktb@gmail.com)  
- Brage Eilertsen — [bragee2002@gmail.com](mailto:bragee2002@gmail.com)
---


> 📅 **Phase 3: Evaluation, Results Discussion, and Initial Paper Drafting**  
> **Duration:** July 14 - July 18
---

## Activite 3: # Evaluation and Paper Drafting

During this stage of the summer school, our activities focused on three main objectives:

## Evaluation of Experimental Results:
- We systematically evaluated the performance of our proposed Supervised Rationale Attention (SRA) method. This included analyzing both interpretability metrics (such as IoU and F1-overlap with human rationales) and fairness metrics (including Subgroup AUC, BPSN AUC, and BNSP AUC) across multilingual datasets.

## Discussion and Interpretation of Results:
- We conducted an in-depth discussion of the experimental findings to identify strengths, potential biases, and areas for further improvement. This involved comparing baseline models with the SRA-enhanced models to understand the impact of incorporating human rationales into attention supervision.

## Initiating Paper Writing:
- We started drafting the paper for submission to ACL. This included outlining the introduction, related work, methodology, and compiling the first version of our results and discussion sections.


---
> 📅 **Phase 2: Initial Research, Benchmarking, Implementation, and Experimentation**  
> **Duration:** July 7 - July 11
---

## Activite 2: # First Round of Experiments: Self-Explainable and Fair Hate Speech Detection
- In this initial experimental phase, we implemented a new innovative method with the objective of improving both model interpretability and bias mitigation in hate speech detection.

## Goals:
- Interpretability: Evaluate whether the propose new method improves interpretability of transformers-based models.

- Bias detection and mitigation: Assess whether the use of human-guided attention reduces model bias toward sensitive social groups (e.g., gender, race, nationality).

## Methodology:
- We trained a BERT-based classifier on benchmark explainable hate speech detection datasets: HateXplain and HateBRXplain in English and Portuguese.

- We introduced automated annotations to flag whether each example contains references to sensitive identity groups (e.g., “woman”, “black”, etc.), enabling targeted bias evaluation.

## Bias metrics included:

 - Subgroup AUC

- BPSN AUC (Background Positive, Subgroup Negative)

- BNSP AUC (Background Negative, Subgroup Positive)

- Generalized Mean of Bias AUCs

## Interpretability metrics included:

- IOU, F1-overlap between model attention and human rationales

- Sufficiency and Comprehensiveness of rationales


---
> 📅 **Phase 1: Preparation & Onboarding**  
> **Duration:** July 1 - July 7
---

## Introduction to the Scope and Objectives of the Project

- **Familiarization with Deep Learning Frameworks:**  
  Getting started with popular deep learning libraries such as [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/), including tools and best practices for building, training, and evaluating deep learning models.

- **Reading Materials and Discussion on Explainability Techniques:**  
  Exploring model interpretability methods like:
  - [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/)
  - [LIME (Local Interpretable Model-agnostic Explanations)](https://lime-ml.readthedocs.io/)
  - [Introduction to Attention mechanisms in deep learning](https://www.ibm.com/think/topics/attention-mechanism)

- **Setting up the Development Environment and Datasets:**  
  Preparing the environment for training and testing, including installation of required libraries, configuration of GPU settings, and loading relevant datasets.

## Activite 1: # Fine-Tuning BERT and Applying LIME & SHAP for Sentiment Analysis  

## Objective

In this activity, you will explore the full pipeline of **fine-tuning a transformer-based model** for **sentiment analysis**, followed by applying **model-agnostic explainability techniques** to interpret its predictions.

This hands-on exercise aims to help you:

- Understand how to adapt a pre-trained BERT model to a new classification task.
- Learn how to use LIME and SHAP to explain model behavior.
- Evaluate what the model learns and how it makes decisions.


## Instructions

### 1. Choose a Sentiment Analysis Dataset

Select a binary dataset available on the [Hugging Face Datasets Hub](https://huggingface.co/datasets) for a sentiment classification task.

You can choose from datasets such as:
- `imdb`
- `financial_phrasebank`
- `sst2`
- `amazon_polarity`
- or any other dataset that suits your interest and has a sentiment-related label.

### 2. Fine-Tune a Pre-trained BERT Model

Use a model from [Hugging Face Transformers](https://huggingface.co/models), such as `bert-base-uncased`, and fine-tune it on your selected dataset.

Steps include:
- Preprocessing and tokenization.
- Splitting the dataset (train/test).
- Fine-tuning the model using Hugging Face `Trainer` or PyTorch Lightning.
- Evaluating accuracy, F1-score, or other relevant metrics.

You can use the provided notebook [activity1_lime&shap_bert_sentiment_analysis.ipynb](https://github.com/franciellevargas/DeepXplain/blob/4d092be2ceb0a4e9b8fe2676dd942dc96451afbe/code/activity1_lime%26shap_bert_sentiment_analysis.ipynb) as a starting point and customize it to your dataset.


### 3. Apply Explainability Methods: LIME and SHAP

After training your model, apply **LIME** and **SHAP** to explain **individual predictions**.

Your task is to:
- Use LIME to highlight the most important words contributing to a given classification.
- Use SHAP (with the `shap.Explainer`) to compute Shapley values for token-level contributions.
- Visualize the explanations and compare how each technique interprets the model's behavior.


### 4. Analyze and Reflect

Include your observations directly in the notebook:
- Which words most influenced the predictions?
- Do LIME and SHAP highlight similar tokens?
- Are there any surprising patterns or signs of bias?
- Did the model behave as you expected?

```bash
pip install transformers datasets torch lime shap scikit-learn matplotlib seaborn

