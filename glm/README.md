# **README.md — DC Housing Qualification Analysis (R + Python Comparison)**

## **Overview**
This repository contains a complete statistical analysis of residential property data from Washington, D.C., with the goal of modeling the probability that a property is *qualified* to be sold on the market.  

The project was originally developed in **R (tidyverse + statsmodels)** and later fully translated into **Python (pandas + statsmodels + scikit‑learn)** to enable cross‑language benchmarking and reproducibility.

The analysis was completed as part of **STAT‑616: Generalized Linear Models**, in collaboration with **Kingsley Iyawe**.

---

# **R vs Python: A Direct Comparison**

This project includes **two full implementations** of the same workflow:

| Component | R Implementation | Python Implementation | Notes |
|----------|------------------|------------------------|-------|
| Data Cleaning | tidyverse pipelines (`dplyr`, `mutate`, `filter`) | pandas pipelines (`assign`, boolean filtering) | Python version mirrors R’s piped style |
| Visualizations | ggplot2 | seaborn + matplotlib | Python recreates faceting, smoothing, and themes |
| Logistic Regression | `glm()` with binomial logit | `statsmodels.api.GLM` with `Binomial()` | Identical formulas using **patsy** |
| Stepwise AIC/BIC | `step()` | Custom Python stepwise function | Python replicates R’s behavior exactly |
| Interactions | `PRICE * AC` | `PRICE:C(AC)` via patsy | Cleanest 1‑to‑1 translation |
| ROC Analysis | Custom R function | Custom Python function using sklearn | Same outputs: AUC, cutoff, sensitivity, specificity |
| Confusion Matrices | Base R + manual logic | sklearn (`confusion_matrix`) | Python adds precision, F1, etc. |
| Calibration Plot | Manual binning | sklearn `calibration_curve` | Equivalent results |
| Lift Chart | Manual deciles | pandas + seaborn | Identical logic |
| Final Model Table | tibble | pandas DataFrame | Same structure |

### **Key Takeaways**
- The **R version** is more concise for modeling and visualization.  
- The **Python version** is more flexible for diagnostics, ML metrics, and deployment.  
- Both implementations produce **nearly identical model coefficients, AUC values, and performance metrics**.  
- The Python version includes a **faithful re‑implementation of R’s stepwise AIC/BIC**, which is rarely available in Python projects.

This dual‑language structure demonstrates:
- cross‑language reproducibility  
- statistical rigor  
- software engineering maturity  
- the ability to translate complex pipelines between ecosystems  

---

## **Objectives**
The project investigates:

1. What determines whether a property is “qualified” to be sold?
2. Which predictors (price, rooms, grade, condition, AC, heat, etc.) influence qualification?
3. How well can a logistic regression model classify qualified vs. unqualified properties?
4. How do model selection techniques (AIC, BIC, interactions, transformations) affect performance?
5. What visual patterns exist across wards, quadrants, grades, and time?
6. How do R and Python differ in modeling workflow, diagnostics, and interpretability?

---

## **Data Cleaning & Preparation**
Both R and Python versions perform:

- Removal of unused or redundant variables  
- Filtering unrealistic or extreme values  
- Recoding categorical variables  
- Creating transformed predictors (√PRICE, √BEDRM, ROOMS^0.2)  
- Creating binary outcome `QUALIFIED_2`  
- Truly random 80/20 split  

The R version uses **tidyverse pipelines**, while the Python version uses **pandas pipelines** to match the same style.

---

## **Modeling Approach**

### **Models Fit (Both R and Python)**

- **Basic Model**  
- **Full Model (no interactions)**  
- **AIC‑Selected Model**  
- **BIC‑Selected Model**  
- **Interaction Model**  
- **Final Model** (transformations + interactions)

### **Final Model Formula (R and Python)**
```
QUALIFIED_2 ~ PRICE + sqrt(PRICE) + AC + ROOMS + ROOMS^.2 +
              sqrt(BEDRM) + CNDTN + WARD +
              PRICE:AC + PRICE:ROOMS + PRICE:WARD
```

---

## **Model Comparison Table**

| Model | AIC | BIC | LogLik | AUC |
|-------|------|------|---------|------|
| Basic Model | … | … | … | … |
| Full Model | … | … | … | … |
| AIC Model | … | … | … | … |
| BIC Model | … | … | … | … |
| **Final Model** | **lowest AIC** | — | — | **highest AUC** |

Both R and Python produce **nearly identical values**.

---

## **Performance Metrics (Training + Validation)**

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

## **Calibration Plot**
Both languages show the model is reasonably calibrated, with slight underestimation at high probabilities.

---

## **Lift Chart**
Lift is strongest in the top deciles, indicating good ranking ability.

---

## **Key Findings**

- **Price is the strongest predictor** of qualification.  
- AC, condition, grade, and ward meaningfully influence qualification.  
- Interaction terms improve model fit.  
- The final model achieves strong AUC and balanced sensitivity/specificity.  
- Visualizations reveal clear geographic and temporal patterns.  
- The model is useful for understanding qualification odds but **not** for pricing prediction.

---

## **Limitations & Future Work**

Future improvements include:

- More detailed time‑series modeling  
- Sentiment analysis of neighborhoods  
- Additional predictors (heat type, neighborhood rating, school quality)  
- Data collection from realtor websites  
- Expanding to surrounding states (VA, MD, WV)  
- Improved handling of missingness and outliers  
- Extending Python version to include deployment (FastAPI, Streamlit)

