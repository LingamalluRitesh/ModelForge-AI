# ModelForge AI — Model Monitoring, Observability & Drift Guide

## 1. Observability Telemetry Overview
ModelForge AI monitors three distinct layers of production operations:
1. **Infrastructure Health**: Pod CPU, RAM, Disk utilization, and worker queue depths.
2. **Application Performance**: Request throughput (RPS), Error rate (4xx/5xx), $p_{50}$, $p_{95}$, and $p_{99}$ latency percentiles.
3. **Machine Learning Model Health**: Prediction confidence distributions, ground truth feedback accuracy, Population Stability Index (PSI), and feature drift.

---

## 2. Statistical Drift Detection Formulas

### 2.1 Population Stability Index (PSI)
$$PSI = \sum_{i=1}^{k} \left( \text{Actual}_i - \text{Expected}_i \right) \times \ln\left( \frac{\text{Actual}_i}{\text{Expected}_i} \right)$$
- $PSI < 0.10$: **Stable** (No significant population change).
- $0.10 \le PSI < 0.25$: **Moderate Drift** (Warning alert).
- $PSI \ge 0.25$: **Significant Drift** (Triggers automated retraining).

### 2.2 Kolmogorov-Smirnov (KS-Test)
Evaluates the maximum difference between cumulative empirical distribution functions:
$$D = \sup_x |F_{\text{baseline}}(x) - F_{\text{target}}(x)|$$
Significant when $p\text{-value} < 0.01$.

### 2.3 Wasserstein Distance (Earth Mover's Distance)
Computes minimal work to transform one probability distribution into another:
$$l_1(u, v) = \int_{-\infty}^{+\infty} |U(x) - V(x)| dx$$
