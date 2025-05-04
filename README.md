### 1. **Original Dataset**
- The original dataset consists of:
  - A categorical feature: `Category1` (values: A–E)
  - Two continuous features: `Value1`, `Value2`

### 2. **Visualising Original Dataset**
- The original dataset was analysed first for understanding the data
  - Observed normal distribution of data
  - Stored in original_data folder
 
### 3. **Generating New Dataset**
- Parameters for new dataset
  - mean, std calculated from original data
  - category-specifc count instead of num_samples

### 4. **Data Analysis**
- For both datasets, the following are generated:
  - Histograms for `Value1` and `Value2` per category
  - Descriptive statistics and skewness saved as JSON

### 5. **Verification of Samples**
- Visual Comparison:
  - Category distribution comparison using bar plots
  - KDE plots for visulisation of continuous distribution of `Value1` and `Value2` 
- Statistical Comparison:
  - **Chi-square test** for categorical variable `Category1`
  - **Kolmogorov–Smirnov (KS) test** for continuous variables `Value1`, `Value2`

### 6. **Statistical Interpretation**
- Null hypothesis: The distribution of two samples is same.
- Alternative hypothesis: The distribution of new sample is different from the original sample.
- If p > 0.05: Fail to reject H₀ -- the distributions are not significantly different.
- If p < 0.05: Reject H₀ -- the distributions are significantly different.

