# Introduction

This is the companion code to the article:


**Mixture of Conditional Gaussian Graphical Models for unlabelled heterogeneous populations in the presence of co-factors**. Thomas Lartigue, Stanley Durrleman, Stéphanie Allassonnière. 

https://hal.inria.fr/hal-02874192/document

If you use elements of this code in your work, please cite this article as reference.

# Abstract

Conditional correlation networks, within Gaussian Graphical Models (GGM) [1], are widely used to describe the direct interactions between the components of a random vector. In the case of an unlabelled Heterogeneous population, Expectation Maximisation (EM) algorithms [2] for Mixtures of GGM have been proposed to estimate both each sub-population's graph and the class labels, see [3] and [4]. However, we argue that, with most real data, class affiliation cannot be described with a Mixture of Gaussian, which mostly groups data points according to their geometrical proximity. In particular, there often exists external co-features whose values affect the features' average value, scattering across the feature space data points belonging to the same sub-population. Additionally, if the co-features' effect on the features is Heterogeneous, then the estimation of this effect cannot be separated from the sub-population identification. In this article, we propose a Mixture of Conditional GGM (CGGM) that subtracts the heterogeneous effects of the co-features to regroup the data points into sub-population corresponding clusters. We develop a penalised EM algorithm to estimate graph-sparse model parameters. We demonstrate on synthetic and real data how this method fulfils its goal and succeeds in identifying the sub-populations where the Mixtures of GGM are disrupted by the effect of the co-features.

# Contents

This repository provides:

- Functions to estimate a Conditional Mixture of Gaussians with a joint-Gaussian Graphical Model structure. The specific penalty used here is Group Graphical Lasso [5].
*Conditional_Gaussian_Graphical_Model_EM.py*

- Functions to estimate a simple Mixture of Gaussians with the same joint-Gaussian Graphical Model structure.
*Gaussian_Graphical_Model_EM.py*

- A toy example in dimension 2 is provided to illustrate the interest of taking into account the heterogeneous effect of observed co-features.
*2D_example.ipynb*


# References

[1] Dempster, Arthur P. "Covariance selection." Biometrics (1972): 157-175.
[2] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the Royal Statistical Society: Series B (Methodological) 39.1 (1977): 1-22.
[3] Gao, Chen, et al. "Estimation of multiple networks in gaussian mixture models." Electronic journal of statistics 10 (2016): 1133.
[4] Hao, Botao, et al. "Simultaneous clustering and estimation of heterogeneous graphical models." Journal of Machine Learning Research (2018).
[5] Danaher, Patrick, Pei Wang, and Daniela M. Witten. "The joint graphical lasso for inverse covariance estimation across multiple classes." Journal of the Royal Statistical Society. Series B, Statistical methodology 76.2 (2014): 373.