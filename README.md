# 🎤 GMM-Based Speaker Identification using the EM Algorithm


## 📌 Overview

This project implements a **speaker identification system** from scratch using **Gaussian Mixture Models (GMMs)** trained via a hand-coded **Expectation-Maximization (EM) algorithm** — without using `sklearn.mixture.GaussianMixture` or any other GMM library. One GMM is trained per speaker on their MFCC features extracted from the **VCTK Corpus**. Test samples are identified by computing the log-likelihood under each speaker's GMM and assigning to the most likely speaker.

---

## 🎯 Objectives

- Download and load the **VCTK Corpus** multi-speaker dataset
- Extract **MFCC features** from raw audio recordings
- Implement the **EM algorithm from scratch** (E-step + M-step)
- Train one **GMM per speaker** using EM on their feature distributions
- Evaluate by computing **log-likelihood** of test samples under each GMM
- Assign each test sample to the speaker with the **highest likelihood**

---

## 📁 Project Structure

```
GMM_Speaker_Identification/
│
├── speaker_identification_using_gaussian_mixture_mode.ipynb   # Main notebook
└── README.md

# Dataset (downloaded via kagglehub at runtime):
# /kaggle/input/vctk-corpus/VCTK-Corpus/VCTK-Corpus/wav48/
#   ├── p225/   ← Speaker directories (~400 .wav files each)
#   ├── p226/
#   └── ...
```

---

## 📦 Dataset

**VCTK Corpus** — Multi-speaker English speech dataset
- **110 speakers**, various English accents
- Each speaker reads ~400 sentences from different sources
- Audio format: 48 kHz WAV files
- Downloaded from Kaggle: [`pratt3000/vctk-corpus`](https://www.kaggle.com/datasets/pratt3000/vctk-corpus)

```python
import kagglehub
kagglehub.dataset_download('pratt3000/vctk-corpus')
```

---

## ⚙️ Pipeline

### 1. Feature Extraction — MFCCs
```
Parameters:
  NUM_SPEAKERS = 5
  N_MFCC       = 13
  SR           = 16000 Hz
```
- Each audio file is loaded, silence-trimmed (`librosa.effects.trim`), and 13 MFCCs extracted
- All utterances per speaker are stacked into a single feature matrix: shape `(total_frames, 13)`

### 2. GMM Initialization
```python
def initialize_gmm(X, n_components=4):
    weights     = uniform (1/K each)
    means       = K random samples from X
    covariances = covariance of X + 1e-6 * I  (regularized)
```

### 3. E-Step — Responsibility Computation
```python
def e_step(X, weights, means, covariances):
    # For each component k:
    gamma[:, k] = weights[k] * N(x | mean_k, cov_k)
    # Normalize across components
    gamma /= gamma.sum(axis=1, keepdims=True)
```
- Computes the posterior probability (responsibility) that each sample belongs to each Gaussian component

### 4. M-Step — Parameter Update
```python
def m_step(X, gamma):
    N_k        = gamma.sum(axis=0)               # effective count per component
    weights    = N_k / n_samples                  # mixing weights
    means      = (gamma.T @ X) / N_k             # weighted means
    covariances = weighted scatter matrix / N_k  # + 1e-6*I regularization
```

### 5. Training Loop
```python
def train_gmm(X, n_components=4, n_iter=20, tol=1e-3):
    # Iterates E-step → M-step until log-likelihood converges
    # Early stopping when |ΔLL| < tol
```
- Trained independently for each speaker using their stacked MFCC features
- `n_components = 4` Gaussian components per speaker GMM

### 6. Prediction — Maximum Likelihood Speaker Assignment
```python
def predict_speaker(test_features, speaker_gmms):
    # For each speaker GMM:
    log_likelihood = Σ log( Σ_k weights[k] * N(x | mean_k, cov_k) )
    # Assign to speaker with highest log-likelihood
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Speakers trained | 5 (from VCTK subset) |
| GMM components per speaker | 4 |
| Feature type | 13 MFCCs |
| EM iterations | Up to 20 (early stop on convergence) |
| Evaluation | Log-likelihood based speaker assignment |
| Test | Correct speaker identified on held-out utterances |

---

## 🔑 Key Implementation Notes

- **EM is implemented from scratch** — no `sklearn.mixture.GaussianMixture` used
- Covariance matrices are **regularized** (`+ 1e-6 * I`) to prevent singularity
- Log-likelihood is computed at each iteration to monitor convergence
- `scipy.stats.multivariate_normal.pdf` is used only for the Gaussian PDF evaluation (not for the EM algorithm itself)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `librosa` | Audio loading, silence trimming, MFCC extraction |
| `numpy` | All matrix operations for EM algorithm |
| `scipy.stats` | Multivariate normal PDF evaluation |
| `matplotlib` | Visualization |
| `kagglehub` | Dataset download |
| `glob`, `os` | File system navigation |

---

## ⚙️ Setup & Installation

```bash
pip install librosa numpy scipy matplotlib kagglehub
```

Run on **Kaggle Notebooks** (recommended — dataset is natively available) or Google Colab with `kagglehub` for dataset download.

```python
# In the notebook, the dataset is loaded as:
import kagglehub
kagglehub.dataset_download('pratt3000/vctk-corpus')
```

---

## 📐 EM Algorithm — Mathematical Summary

**E-Step:**

$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M-Step:**

$$N_k = \sum_n \gamma_{nk}, \quad \pi_k = \frac{N_k}{N}, \quad \boldsymbol{\mu}_k = \frac{1}{N_k}\sum_n \gamma_{nk} \mathbf{x}_n$$

$$\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_n \gamma_{nk}(\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top$$

---
