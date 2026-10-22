# Recorded results and protocol limits

All accuracies are percentages. Means and sample standard deviations use three SSL seeds (0, 1, 2), with one fixed split (1729).

## Frozen-encoder kNN accuracy

| Method | CIFAR-10 | CIFAR-100 | Tiny ImageNet | Imagenette |
|---|---:|---:|---:|---:|
| CSR | 76.02 ± 0.36 | 44.04 ± 0.12 | 34.05 ± 0.41 | 81.41 ± 0.67 |
| MMCR control | 73.53 ± 0.06 | 41.64 ± 0.13 | 29.32 ± 0.12 | 78.46 ± 0.12 |
| VICReg | 74.70 ± 0.37 | 36.42 ± 0.07 | 33.12 ± 0.20 | 81.10 ± 0.25 |

The README reports the corresponding linear accuracies. Do not pool the 30-epoch CIFAR and 100-epoch ImageNet-subset budgets.

## CIFAR-100 training ablations

| Setting | Test linear | Test kNN | True-pair cosine | Unit-projector entropy rank |
|---|---:|---:|---:|---:|
| CSR, κ=10 | 49.69 ± 0.10 | 44.04 ± 0.12 | 0.723 | 244.4 |
| Normalize centroid again | 11.41 ± 0.22 | 9.84 ± 0.30 | 0.020 | 247.0 |
| Shuffle training pairs | 14.65 ± 0.52 | 12.21 ± 0.55 | 0.040 | 244.8 |
| CSR, κ=1 | 46.29 ± 0.18 | 39.50 ± 1.13 | 0.660 | 246.3 |
| CSR, κ=100 | 49.07 ± 0.19 | 40.33 ± 0.98 | 0.817 | 91.6 |

Geometry is measured on correctly paired held-out internal-validation augmentations for every condition, including shuffled-pair training. Entropy rank uses normalized singular values of centered unit projector features. It is not the ridge effective dimension in the training loss. No condition triggers the recorded collapse flags, although two interventions yield poor representations.

All conditions use 30 epochs, 45,000 training images, 5,000 internal-validation images, batch 256, and the same optimization recipe. The κ=10 control repeats the original three seeds and is not pooled as three new independent samples. Renormalization also changes gradient conditioning, so the intervention does not isolate every causal pathway.

## Limitations

The baseline learning rates and other settings were shared, not independently tuned. CIFAR gradient clipping affects VICReg substantially more often than CSR/MMCR. The extension removes clipping, but also changes dataset and training budget, so it is not a controlled test of clipping alone.

Three seeds on one split do not establish universal superiority or convergence. Paired intervals in the JSON are exploratory, unadjusted for selection/multiple comparisons. The method retains its originally selected κ=10; the additional test scores are not used to select a replacement setting.

All final extension and ablation conditions are retained in the accompanying JSON, including results favoring VICReg. ImageNet-subset `heldout` scores correspond to official validation images reserved from tuning, not the unlabeled official Tiny ImageNet test split.
