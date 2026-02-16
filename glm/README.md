# **README.md — DC Housing Qualification Analysis (R + Python Comparison)**

## **Overview**
This repository contains a complete statistical analysis of residential property data from Washington, D.C., with the goal of modeling the probability that a property is *qualified* to be sold on the market.

The project was originally developed in **R (tidyverse + glm)** and later fully translated into **Python (pandas + statsmodels + scikit‑learn)** to enable cross‑language benchmarking and reproducibility. Implementing the full workflow in both languages demonstrates conceptual mastery, statistical rigor, and software engineering versatility.

The analysis was completed as part of **STAT‑616: Generalized Linear Models**, in collaboration with **Kingsley Iyawe**.

---

# **Analytical Summary of the Work**

## **Analytical Overview**
This project investigates the factors that determine whether a residential property in Washington, D.C. is *qualified* to be sold on the market. The analysis follows a structured statistical workflow:

### **1. Data Cleaning & Preprocessing**
The raw dataset contained inconsistencies, missing values, and structural issues. Cleaning steps included:

- Removing non‑informative or redundant variables  
- Filtering unrealistic values (e.g., PRICE < \$10,000 or > \$10M)  
- Recoding categorical variables (AC, GRADE, CNDTN)  
- Engineering transformed predictors (√PRICE, √BEDRM, ROOMS^0.2)  
- Creating a binary outcome variable `QUALIFIED_2`  
- Splitting into training and validation sets  

These steps ensured the modeling dataset was statistically valid and interpretable.

### **2. Exploratory Data Analysis (EDA)**
Visualizations revealed strong structural patterns:

- **Price varies dramatically by Ward and Quadrant**, with NW and Wards 2–3 consistently higher.  
- **Rooms, bedrooms, kitchens, and stories** show nonlinear relationships with price.  
- **Condition and grade** strongly correlate with both price and qualification.  
- **Time trends** show increasing prices over the years.  

These insights guided the choice of transformations and interactions in the model.

### **3. Model Building**
Multiple logistic regression models were fit:

- Basic model (PRICE only)  
- Full model (all predictors)  
- AIC‑selected model  
- BIC‑selected model  
- Interaction model  
- Final model with transformations and interactions  

The final model was chosen based on:

- Lower AIC  
- Stronger AUC  
- Better interpretability  
- More stable coefficients  

### **4. Model Interpretation**
Key findings:

- **Price is the strongest predictor** of qualification.  
- **AC, condition, grade, and ward** meaningfully influence qualification.  
- **Interactions** (e.g., PRICE × AC) capture important nonlinear effects.  
- **Transformations** improve model fit and residual behavior.  

The model estimates the **odds** that a property is qualified, not the price itself.

### **5. Model Evaluation**
Using both training and validation sets:

- ROC curves show strong separation  
- AUC values indicate good predictive ability  
- Confusion matrices show balanced sensitivity and specificity  
- Calibration plots show reasonable probability alignment  
- Lift charts show strong ranking performance in top deciles  

The model generalizes well and is statistically sound.

### **6. Limitations**
- The meaning of “qualification” in the original dataset is unclear  
- Some variables (e.g., heat type) lack granularity  
- The dataset is geographically limited to D.C.  
- A linear model would be more appropriate for predicting price  

### **7. Future Work**
- More detailed time‑series modeling  
- Sentiment analysis of neighborhoods  
- Additional predictors (school quality, neighborhood ratings)  
- Multi‑state data integration  
- Improved missing data handling  
- Deployment as a Shiny or Python dashboard  

---

# **R vs Python: A Direct Comparison**

This project includes **two complete implementations** of the same statistical workflow — one in **R** and one in **Python**. Re‑creating the entire pipeline in both languages ensures that the analysis is conceptually sound, reproducible, and not dependent on a single software ecosystem.

| Component | R Implementation | Python Implementation | Notes |
|----------|------------------|------------------------|-------|
| Data Cleaning | tidyverse pipelines (`dplyr`, `mutate`, `filter`) | pandas pipelines (`assign`, boolean filtering) | Python mirrors R’s piped style |
| Visualizations | ggplot2 | seaborn + matplotlib | Python recreates faceting, smoothing, themes |
| Logistic Regression | `glm()` (binomial logit) | `statsmodels.GLM` (Binomial) | Identical formulas via **patsy** |
| Stepwise AIC/BIC | `step()` | Custom Python stepwise function | Python replicates R’s behavior |
| Interactions | `PRICE * AC` | `PRICE:C(AC)` | 1‑to‑1 translation |
| ROC Analysis | Custom R function | Custom Python function using sklearn | Same outputs: AUC, cutoff, sensitivity |
| Confusion Matrix | Base R | sklearn (`confusion_matrix`) | Python adds precision, F1 |
| Calibration Plot | Manual binning | sklearn `calibration_curve` | Equivalent results |
| Lift Chart | Manual deciles | pandas + seaborn | Identical logic |
| Final Model Table | tibble | pandas DataFrame | Same structure |

### **Key Takeaways**
- **R is more concise** for modeling and visualization.  
- **Python is more flexible** for diagnostics, ML metrics, and deployment.  
- Both implementations produce **nearly identical coefficients, AUC values, and performance metrics**.  
- The Python version includes a **faithful re‑implementation of R’s stepwise AIC/BIC**, which is rarely available in Python projects.  

This dual‑language structure demonstrates:

- cross‑language reproducibility  
- statistical rigor  
- software engineering maturity  
- the ability to translate complex pipelines between ecosystems  

---

# **Performance Metrics (Training + Validation)**

Both implementations compute:

- Accuracy  
- Sensitivity  
- Specificity  
- Precision  
- F1 Score  
- AUC  
- Optimal cutoff  
- Confusion matrices  

Python uses sklearn for metric computation; R uses manual logic.

---

# **Calibration Plot**
Both languages show the model is reasonably calibrated, with slight underestimation at high predicted probabilities.

---

# **Lift Chart**
Lift is strongest in the top deciles, indicating strong ranking ability and good separation between qualified and unqualified properties.

---

# **Key Findings**
- **Price is the strongest predictor** of qualification.  
- AC, condition, grade, and ward meaningfully influence qualification.  
- Interaction terms improve model fit.  
- The final model achieves strong AUC and balanced sensitivity/specificity.  
- Visualizations reveal clear geographic and temporal patterns.  
- The model is useful for understanding qualification odds but **not** for predicting price.

---

# **Why Dual‑Language Modeling Matters**

Working in both **R** and **Python** is more than a stylistic choice — it reflects a deeper level of statistical and computational fluency. Each language brings its own strengths to the table, and translating an entire modeling workflow across ecosystems demonstrates that the analysis is driven by *concepts*, not syntax.

R excels at statistical modeling, formula notation, and expressive data visualization.  
Python shines in machine learning tooling, extensibility, and production‑grade workflows.

By implementing the full pipeline — cleaning, feature engineering, logistic regression, stepwise AIC/BIC selection, ROC analysis, calibration, lift charts, and performance evaluation — in **both** languages, this project shows:

- **Reproducibility:** Results are not tied to a single toolchain.  
- **Cross‑validation of logic:** If both languages produce the same conclusions, the modeling decisions are sound.  
- **Technical versatility:** The ability to move seamlessly between R and Python is a core skill in modern data science and biostatistics.  
- **Deployment readiness:** R is ideal for exploratory modeling; Python is ideal for scaling, automation, and integration with applications.  
- **Conceptual mastery:** Rewriting stepwise AIC/BIC selection, interaction terms, and custom ROC logic in Python requires understanding the underlying mathematics, not just copying code.

Dual‑language modeling signals that the analysis is **robust**, **portable**, and **professionally engineered** — and that the analyst can operate effectively in any environment, whether academic, clinical, or industry‑focused.
