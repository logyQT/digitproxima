# DigitProxima — Handwritten Digit Recognition with k‑NN and Tkinter

An interactive OCR system built on a **k‑Nearest Neighbours** classifier with a
graphical user interface. The user draws a digit on a canvas and the system runs
the full pipeline: image preprocessing, classification and visualisation of the
result.

## Table of contents

- [System requirements](#system-requirements)
- [Installation](#installation)
- [Running the app](#running-the-app)
- [Dataset](#dataset)
- [System architecture](#system-architecture)
  - [User interface](#user-interface)
  - [Image preprocessing](#image-preprocessing)
  - [Classification](#classification)
  - [Visualisations](#visualisations)
- [Project structure](#project-structure)
- [Possible directions for development](#possible-directions-for-development)
- [Summary](#summary)

---

## System requirements

- **Python** 3.10–3.12
- **Tkinter** (part of the standard Python distribution)

### Dependencies

| Library        | Version |
| -------------- | ------- |
| `numpy`        | 1.26.4  |
| `Pillow`       | 10.2.0  |
| `scikit-learn` | 1.4.2   |
| `matplotlib`   | 3.8.3   |

---

## Installation

```bash
pip install -r requirements.txt
```

## Running the app

```bash
python main.py
```

1. Draw a digit (0–9) on the white canvas with the mouse.
2. Enter the **k** parameter (the number of nearest neighbours); the field is
   pre-filled with `41`.
3. Click **Predict** to run the classification.
4. Click **Clear** to wipe the canvas and start again.

---

## Dataset

The system uses the **Digits** dataset from `sklearn.datasets`:

| Property                | Value        |
| ----------------------- | ------------ |
| Number of samples       | 1797         |
| Image resolution        | 8×8 pixels   |
| Pixel value range       | 0–16         |
| Number of classes       | 10 (0–9)     |

The dataset is the reference training set for the k‑NN classifier.

---

## System architecture

### User interface

Built on **Tkinter**, the window has three main panels:

```
┌──────────────────────────────────────────────────────┐
│                     KNN OCR                          │
├───────────┬───────────────────────┬──────────────────┤
│           │  Average of k         │                  │
│  Drawing  │  neighbors            │  k-NN voting     │
│  canvas   │  Composite diff       │  histogram       │
│  300×300  │  32×32 preview        │                  │
│           │                       │                  │
├───────────┴───────────────────────┴──────────────────┤
│  [k value: ___]  [Clear]  [Predict]                  │
│  Prediction: ?                                       │
└──────────────────────────────────────────────────────┘
```

- **Drawing panel** — a 300×300 px area for entering a digit by hand.
- **Visualisation panel** — three plots: the rescaled 32×32 image, the mean of
  the _k_ nearest neighbours' images, and the difference map between the drawing
  and that mean.
- **Voting panel** — a histogram of how the labels of the _k_ nearest neighbours
  are distributed across the classes.

### Image preprocessing

The input pipeline:

```
300×300 drawing
       ↓
  Resize (Lanczos) → 32×32
       ↓
  Colour inversion
       ↓
  Thresholding (threshold = 50)
       ↓
  Reduction 32×32 → 8×8 (sum over each 4×4 block)
       ↓
  Flattening → feature vector (64 dimensions)
```

Each stage brings the input into the same format as the Digits dataset. Thresholding
produces a binary image, so each 4×4 block sums to a value in the 0–16 range the
dataset uses.

### Classification

| Parameter             | Value              |
| --------------------- | ------------------ |
| Algorithm             | k‑NN               |
| Distance metric       | Manhattan (L1)     |
| Search method         | brute force        |
| Number of neighbours (k) | set by the user |

### Visualisations

The system produces three key visualisations:

1. **Mean of the nearest neighbours**

   $$A = \frac{1}{k} \sum_{i=1}^{k} N_i$$

   The image is the arithmetic mean of the _k_ nearest neighbours in 32×32 space.

2. **Difference map**

   $$D = |A - I|$$

   The absolute difference between the user's drawing and the neighbours' mean —
   it highlights the pixels that deviate most from the pattern.

3. **Voting histogram**

   The distribution of labels among the _k_ nearest neighbours — it gives a sense
   of how confident the classification is.

---

## Project structure

```
.
├── main.py              # Application source code
├── requirements.txt     # Dependency list
└── README.md            # Project documentation
```

---

## Possible directions for development

- Additional classifiers (SVM, neural networks)
- Preprocessing filters (e.g. Gaussian smoothing)
- Training on user-supplied data
- Exporting classification reports
- Integration with deep learning models

---

## Summary

The project is a complete, interactive OCR system based on a k‑NN classifier.
Thanks to its visualisations, the user can analyse the classification process at
the level of individual pixels, which makes the system a valuable teaching and
experimentation tool. The application combines classical machine learning methods
with an intuitive graphical interface, making the properties of the k‑NN algorithm
practical to study.
