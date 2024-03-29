# t(s)SVDD
_A Python package for SVDD with time-series kernels._

[![][docs-img]][docs-url][![Python package](https://github.com/thiessenh/tsvdd/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/thiessenh/tsvdd/actions/workflows/python-package.yml)

This package implements [SVDD](https://en.wikipedia.org/wiki/One-class_classification) suitable for time-series anomaly detection. The package was developed as part of my master's thesis _“Detecting Outlying Time Series with Global Alignment Kernels”_.

## Getting started
### Installation
```bash
	pip install git+https://github.com/thiessenh/tsvdd.git@main
```
### Usage
```python
svdd = SVDD(nu=0.05, sigma='auto') # set C according to 5% outlier, determine \sigma automatically with heuristic
svdd.fit(X) # train SVDD
y_pred = svdd.predict(X) # predict, a -1 label indicates outliers
```
See `notebooks/example_notebook.ipynb` for a more detailed illustration.
## Overview
To tackle the challenging problem of outlier detection of time series data, we propose the combination of SVDD and TGAK as kernel function.

### SVDD
SVDD [1] is a SOTA outlier detector that comes with a set of advantages. As it is unsupervised, it does not need labeled data. Also, SVDD is a convex optimization problem and thus has a globally optimal solution.
### Global Alignment Kernels
Using GA kernels as kernel function avoids time series transformations. Contrary to other time-series kernels, GA kernels are positive definite.

### Choosing parameters <img src="https://render.githubusercontent.com/render/math?math=C">, <img src="https://render.githubusercontent.com/render/math?math=\sigma">, and <img src="https://render.githubusercontent.com/render/math?math=T">.

#### Parameter <img src="https://render.githubusercontent.com/render/math?math=C">
C is SVDD’s trade-off parameter, also referred to as cost parameter. It allows us to set the ratio of training data regarded as outliers.
```math
C = \frac{1}{\nu |X|}
```

#### Parameter <img src="https://render.githubusercontent.com/render/math?math=\sigma">
GA kernels employ a modified Gaussian kernel. Therefore, setting the bandwidth <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is specific to the problem at hand. Cuturi et al. [2] suggest to set σ depending on the complexity and length of the time series x and y:
```math
\sigma = {0.1, 1, 10} * median\|x - y\| \sqrt{median x}
```

#### Parameter <img src="https://render.githubusercontent.com/render/math?math=T">
The triangular parameter T restricts the number of valid alignments. Larger T values consider more alignments, whereas smaller values consider fewer alignments. 
When two time series’ lengths differ by more than T − 1, the kernel value is 0. Setting T = 0 considers all alignments. Cuturi et al. [2] suggest setting T to a multiple of the median time series length, such as 0.2 or 0.5.

## Dependencies

- Python >=3.6
- pandas
- numpy 
- dtaidistance
- tsfresh
- sklearn

## Acknowledgements
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [Triangular Global Alignment Kernels](https://marcocuturi.net/GA.html)
- [scikit-learn](https://scikit-learn.org/)


[docs-img]: https://img.shields.io/badge/docs-master-blue.svg
[docs-url]: https://thiessenh.github.io/tsvdd/

## References 
[1] Tax, D. M., & Duin, R. P. (2004). Support vector data description. Machine learning, 54(1), 45-66.

[2] Cuturi, M. (2011). Fast global alignment kernels. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 929-936).
