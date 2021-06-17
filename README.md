# t(s)SVDD package
_A Python package for SVDD with time series kernels._

[![][docs-img]][docs-url]

This package implements [SVDD](https://en.wikipedia.org/wiki/One-class_classification) suitable for time series anomaly detection. The package was developed as part of my master's thesis _“Detecting Outlying Time Series with Global Alignment Kernels”_.

## Installation
```bash
	pip install git+https://github.com/thiessenh/tsvdd.git@main
```
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
