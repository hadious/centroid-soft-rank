# Mathematical notes

For normalized view rows `u_i,v_i`, let `c_i=(u_i+v_i)/2`, `G=CᵀC/B`, `r=min(B,d)`, and `τ=κ/r`. CSR minimizes

$$\mathcal L=1-\frac{1+\kappa}{r}\sum_{j=1}^r\frac{\lambda_j}{\lambda_j+\tau}.$$

Eigenvalues are padded with zeros to length `r`. `G` is an uncentered second-moment matrix, not a centered covariance matrix.

For unit rows, centroid energy satisfies

$$T=\operatorname{tr}(G)=1-\frac{\|U-V\|_F^2}{4B}.$$

Writing `a=(1+κ)/r` and `q=T/r`, the original single objective has the exact decomposition

$$\mathcal L=\frac{\kappa(1-T)}{T+\kappa}+\frac{a\tau}{(q+\tau)^2}\sum_j\frac{(\lambda_j-q)^2}{\lambda_j+\tau}.$$

The first term depends on agreement; the second is a nonnegative spectral-dispersion penalty. For rows below the normalization floor, the first term also reflects their norm deficit. This identity does not add auxiliary losses.

A loss at most `η<1` bounds mean squared normalized-view distance by `4η(1+κ)/(κ+η)` and implies rank at least `r(1−η)/(1+κη)`. These are batch-output statements, not guarantees of semantic recovery or convergence. Constant aligned outputs and antipodal outputs can be stationary after normalization.

Renormalizing each centroid removes the same alignment certificate. For `B=d≥2`, take cyclic orthonormal basis vectors and set `u_i=√s e_i+√(1−s)e_(i+1)` and `v_i=√s e_i−√(1−s)e_(i+1)`. The normalized centroids are `e_i`, giving zero modified loss, while mean pair disagreement is `4(1−s)`. The original loss is `κ(1−s)/(s+κ)`. With the implementation's centroid-normalization floor, this example applies when `√s≥10⁻⁶`.

Run `python check_theory.py` for 278 float64 checks of the identities, bounds, gradient, and counterexamples. Numerical checks complement, rather than replace, the algebra.
