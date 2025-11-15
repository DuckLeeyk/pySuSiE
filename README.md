# pySuSiE

Python implementation of the SuSiE (Sum of Single Effects) and the SuSiE-ss (SuSiE with summary statistics) model for Bayesian variable selection and fine-mapping.

## Overview
pySuSiE provides:
- NumPy/SciPy implementation of SuSiE (individual-level data).
- Summary statistics implementation of SuSiE-ss from z-scores, LD matrices, and sample sizes.
- Posterior inclusion probabilities (PIPs), and credible set (CS) construction with purity metrics.
- Optional prior and residual variance estimation.

## Installation

```bash
# Create and activate a fresh environment
conda create -n pysusie python=3.10 -y
conda activate pysusie

# Clone the repository
git clone https://github.com/your-user/pySuSiE.git
cd pySuSiE

# Install via pip (editable mode for development)
pip install -e .
```

## Requirements
Core dependencies:
- Python >= 3.9
- numpy
- scipy

All are declared in `pyproject.toml` and `requirements.txt`.

## Quick Start
```python
import numpy as np
from pysusie import SuSiE, SuSiE_SS

np.random.seed(0)

# Individual-level data example
n, p = 500, 100
X = np.random.randn(n, p)
true_b = np.zeros(p); true_b[[5, 42]] = [0.8, -1.2]
y = X @ true_b + 0.5 * np.random.randn(n)

model = SuSiE(L=10).fit(X, y)
print("PIP:", model.pip)
print("Credible sets:", model.sets)

# Summary statistics example
Xc = X - X.mean(axis=0, keepdims=True)
yc = y - y.mean()
sdX = Xc.std(axis=0, ddof=1)
sdY = yc.std(ddof=1)
z = np.sqrt(n) * (Xc.T @ yc) / ((n - 1) * sdX * sdY)
R = np.corrcoef(X, rowvar=False)

ss = SuSiE_SS(L=10).fit(z=z, R=R, N=n, coverage=0.95, min_abs_corr=0.5)
print("PIP (SS):", ss.pip)
print("Credible sets (SS):", ss.sets)
```

## Citation
If you use this package in academic work, please cite the original SuSiE papers:

- Wang, G., Sarkar, A., Carbonetto, P., Stephens, M. (2020). A Simple New Approach to Variable Selection in Regression, with Application to Genetic Fine Mapping. Journal of the Royal Statistical Society Series B: Statistical Methodology, 82(5), 1273–1300. https://doi.org/10.1111/rssb.12388
- Zou, Y., Carbonetto, P., Wang, G., Stephens, M. (2022). Fine-mapping from summary data with the “Sum of Single Effects” model. PLOS Genetics, 18(7): e1010299. https://doi.org/10.1371/journal.pgen.1010299
