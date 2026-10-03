# RAFS: Riemannian Adaptive Fractional Spline Framework for Ultrasound Super-Resolution

Official supplementary codebase and reproducibility scripts for the manuscript submitted to *Signal, Image and Video Processing* (SIVP).

**Author:** Mahmoud Saeedi Kelishami  
**Affiliation:** Department of Mathematics, Institute of Biosocial and Quantum Science and Technologies, Rasht Branch, Islamic Azad University, Rasht, Iran  
**Contact:** saeedi@iau.ac.ir  

---

## 📌 Repository Overview

This repository contains the numerical implementations of the **Riemannian Adaptive Fractional Spline (RAFS)** super-resolution framework ($\times 4$) designed for edge-preserving and speckle-aware ultrasound imaging, alongside baseline evaluation pipelines (SRCNN).

The included scripts provide both spatial and frequency-domain accelerations, exact fractional B-spline kernel formulations, Riemannian metric tensor computations, and standardized benchmarking scripts across single scans and clinical cohorts.

---

## 📂 Supplementary Codebase & Script Descriptions

The repository includes five primary Python implementations:

### 1. `rafs_v4_corrected_clean.py`
- **Role:** Foundational / Canonical Implementation of RAFS.
- **Description:** Implements the mathematically rigorous continuous-order fractional B-spline operator over Riemannian manifolds. It computes the local metric tensor components ($g_{11}, g_{12}, g_{22}$) from the smoothed gradient structure tensor, extracts the Riemannian volume element $\sqrt{\det g}$, and adaptively modulates the fractional interpolation order $\alpha(x) \in [1.1, 2.8]$. It performs direct spatial-domain continuous convolution.

### 2. `rafs_v4_optimized_fft.py`
- **Role:** Fast Frequency-Domain Acceleration.
- **Description:** Accelerates the continuous fractional spline filtering stage by transforming spatial 2D convolutions into the spectral domain using Fast Fourier Transforms (FFT). By leveraging the convolution theorem:
  $$\mathcal{F}\{f * \beta^\alpha\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{\beta^\alpha\}$$
  this script substantially reduces runtime complexity on high-resolution image matrices and multi-frame ultrasound cine loops without degrading numerical accuracy.

### 3. `rafs_v4_corrected_clean_optimized.py`
- **Role:** Production & Clinical Deployment Pipeline (Recommended).
- **Description:** Combines the strict Riemannian boundary conditions of the canonical model with vectorized spatial separable filtering and localized metric caching. It includes optimized grid coordinate mapping, clip safeguards, and efficient memory management tailored for real-time ultrasound workstations and resource-constrained environments (e.g., standard clinical PCs without dedicated GPUs).

### 4. `evaluate_srcnn.py`
- **Role:** Single-Case Deep Learning Benchmark.
- **Description:** Evaluates a pre-trained Super-Resolution Convolutional Neural Network (SRCNN, $\times 4$) on individual clinical ultrasound cases (e.g., `ALWI_000`). It computes point-wise Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and records execution latency for direct comparison against RAFS.

### 5. `evaluate_srcnn_cohort.py`
- **Role:** Cohort-Scale Statistical Validation & Benchmarking.
- **Description:** Automates batch evaluation across the entire clinical dataset/cohort (e.g., `Breast image and its masks`). It generates comparative paired statistical metrics (PSNR, SSIM, execution latency) between RAFS, SRCNN, and traditional interpolation techniques (Bilinear, Bicubic, Quadratic B-spline), producing reproducible CSV summaries for journal publication.

---

## ⚙️ Requirements & Dependencies

The algorithms are designed to run efficiently on standard CPU hardware without requiring GPU acceleration.

- Python $\ge 3.8$
- NumPy
- SciPy
- Pillow (PIL)
- PyTorch (optional, required only for running the deep learning comparison scripts `evaluate_srcnn*.py`)

Install necessary dependencies:
```bash
pip install numpy scipy pillow torch torchvision
