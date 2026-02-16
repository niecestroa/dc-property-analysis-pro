"""
LAST EDITTED: February 2, 2026
DC Housing Qualification Analysis — Full Python Translation (Option A)
Mirrors the original R GLM project using pandas, statsmodels, sklearn, seaborn, matplotlib.
Uses patsy formulas for logistic regression and stepwise AIC/BIC selection.
"""

# =========================
# Imports
# =========================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from patsy import dmatrices

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# For nicer tables if desired
pd.set_option("display.float_format", lambda x: f"{x:0.4f}")


# =========================
# 1. Load Data
# =========================

# Adjust path as needed
data_path = r"~/Documents/STAT 616 Generalizd Linear Models/GLM Project/Data/DC_Properties.xlsx"
DC_Properties = pd.read_excel(data_path)

print(DC_Properties.info())
print(DC_Properties.describe(include="all"))


# =========================
# 2. Cleaning / Feature Engineering
# =========================

# Visualization dataset equivalent to dcproperty in R
dcproperty = (
    DC_Properties
    .drop(columns=["CMPLX_NUM", "LIVING_GBA", "SALE_NUM", "GIS_LAST_MOD_DTTM"], errors="ignore")
    .query("PRICE > 10000 and PRICE < 10000000")
    .query("HEAT != 'No Data'")
    .query("CNDTN != 'No Data' and CNDTN != 'Default'")
    .query("STRUCT != 'Default'")
    .query("GRADE != ' No Data'")
    .query("STYLE != 'Default'")
    .query("KITCHENS <= 10")
    .query("ROOMS < 26")
    .query("BEDRM < 20")
    .query("STORIES < 100")
    .query("BATHRM > 0")
    .query("HF_BATHRM > 0")
    .copy()
)

# QUALIFIED_2
dcproperty["QUALIFIED_2"] = np.where(dcproperty["QUALIFIED"] == "Q", 1, 0)

# AC recode
dcproperty["AC"] = dcproperty["AC"].replace({"0": "N"})

# GRADE recode
dcproperty["GRADE"] = dcproperty["GRADE"].replace(
    {"Exceptional-A": "Exceptional",
     "Exceptional-B": "Exceptional",
     "Exceptional-C": "Exceptional",
     "Exceptional-D": "Exceptional"}
)

dcproperty = dcproperty.dropna()
print("dcproperty shape:", dcproperty.shape)

# Final modeling dataset equivalent to DC_Final in R
DC_Final = (
    DC_Properties
    .drop(
        columns=[
            "NUM_UNITS", "YR_RMDL", "SALEDATE", "GBA", "STRUCT", "EXTWALL", "ROOF",
            "INTWALL", "CMPLX_NUM", "LIVING_GBA", "FULLADDRESS", "CITY", "STATE",
            "NATIONALGRID", "ASSESSMENT_SUBNBHD", "CENSUS_BLOCK", "SALE_NUM",
            "GIS_LAST_MOD_DTTM"
        ],
        errors="ignore"
    )
    .query("CNDTN != 'No Data' and CNDTN != 'Default'")
    .query("GRADE != ' No Data'")
    .query("STYLE != 'Default'")
    .query("PRICE > 10000 and PRICE < 10000000")
    .query("FIREPLACES < 8")
    .query("KITCHENS <= 10")
    .query("ROOMS < 26")
    .query("BEDRM < 20")
    .query("STORIES < 100")
    .query("BATHRM > 0")
    .query("HF_BATHRM > 0")
    .copy()
)

DC_Final["QUALIFIED_2"] = np.where(DC_Final["QUALIFIED"] == "Q", 1, 0)
DC_Final["AC"] = DC_Final["AC"].replace({"0": "N"})
DC_Final = DC_Final.dropna()

print("DC_Final shape:", DC_Final.shape)
print(DC_Final.describe(include="all"))

# =========================
# 3. Train / Validation Split (Truly Random)
# =========================

DC_Final = DC_Final.reset_index(drop=True)
DC_Final[".id"] = np.arange(len(DC_Final))

train_df, valid_df = train_test_split(DC_Final, test_size=0.2, random_state=123, shuffle=True)

Final_T = train_df.copy()
Final_V = valid_df.copy()

# Factor-like columns
for col in ["AC", "STYLE", "CNDTN", "WARD"]:
    if col in Final_T.columns:
        Final_T[col] = Final_T[col].astype("category")
        Final_V[col] = Final_V[col].astype("category")

Final_T["QUALIFIED_2"] = Final_T["QUALIFIED_2"].astype(int)
Final_V["QUALIFIED_2"] = Final_V["QUALIFIED_2"].astype(int)

# For modeling convenience
Final_T2 = Final_T.copy()
Final_V2 = Final_V.copy()
Final_V2["y"] = Final_V2["QUALIFIED_2"]


# =========================
# 4. Logistic Regression Models (statsmodels + patsy)
# =========================

# Basic model
formula_basic = "QUALIFIED_2 ~ PRICE"
basic_model = smf.glm(formula=formula_basic, data=Final_T2,
                      family=sm.families.Binomial()).fit()
print(basic_model.summary())

# Full model (no interactions) — mirror model3
formula_full = """
QUALIFIED_2 ~ PRICE + BATHRM + HF_BATHRM + C(AC) + ROOMS + BEDRM + STORIES +
               C(STYLE) + C(CNDTN) + KITCHENS + FIREPLACES + C(WARD)
"""
model3 = smf.glm(formula=formula_full, data=Final_T2,
                 family=sm.families.Binomial()).fit()
print(model3.summary())


# =========================
# 5. Stepwise AIC/BIC Selection (Custom)
# =========================

def stepwise_selection(data, response, predictors, direction="both", criterion="AIC"):
    """
    Simple stepwise selection using statsmodels GLM (binomial).
    predictors: list of strings (each is a term in the formula).
    direction: 'forward', 'backward', 'both'
    criterion: 'AIC' or 'BIC'
    """
    def fit_model(predictor_list):
        if len(predictor_list) == 0:
            formula = f"{response} ~ 1"
        else:
            formula = f"{response} ~ " + " + ".join(predictor_list)
        model = smf.glm(formula=formula, data=data,
                        family=sm.families.Binomial()).fit()
        return model

    included = []
    best_model = fit_model(included)
    best_score = getattr(best_model, criterion.lower())

    changed = True
    while changed:
        changed = False

        # Forward step
        if direction in ["forward", "both"]:
            excluded = list(set(predictors) - set(included))
            new_scores = []
            for new_var in excluded:
                model = fit_model(included + [new_var])
                score = getattr(model, criterion.lower())
                new_scores.append((score, new_var, model))
            if new_scores:
                best_new_score, best_new_var, best_new_model = sorted(new_scores, key=lambda x: x[0])[0]
                if best_new_score < best_score:
                    included.append(best_new_var)
                    best_score = best_new_score
                    best_model = best_new_model
                    changed = True

        # Backward step
        if direction in ["backward", "both"] and included:
            new_scores = []
            for var in included:
                trial = [v for v in included if v != var]
                model = fit_model(trial)
                score = getattr(model, criterion.lower())
                new_scores.append((score, var, model))
            if new_scores:
                best_new_score, worst_var, best_new_model = sorted(new_scores, key=lambda x: x[0])[0]
                if best_new_score < best_score:
                    included.remove(worst_var)
                    best_score = best_new_score
                    best_model = best_new_model
                    changed = True

    return best_model, included


# Predictors for stepwise (no interactions here, like model3)
predictors_no_inter = [
    "PRICE", "BATHRM", "HF_BATHRM", "C(AC)", "ROOMS", "BEDRM", "STORIES",
    "C(STYLE)", "C(CNDTN)", "KITCHENS", "FIREPLACES", "C(WARD)"
]

# AIC-based stepwise
model_aic, included_aic = stepwise_selection(
    data=Final_T2,
    response="QUALIFIED_2",
    predictors=predictors_no_inter,
    direction="both",
    criterion="AIC"
)
print("AIC-selected predictors:", included_aic)
print(model_aic.summary())

# BIC-based stepwise
model_bic, included_bic = stepwise_selection(
    data=Final_T2,
    response="QUALIFIED_2",
    predictors=predictors_no_inter,
    direction="both",
    criterion="BIC"
)
print("BIC-selected predictors:", included_bic)
print(model_bic.summary())


# =========================
# 6. Interaction Model & Final Model
# =========================

# Interaction model (similar to model_inter in R)
formula_inter = """
QUALIFIED_2 ~ PRICE + C(AC) + ROOMS + BEDRM + C(CNDTN) + C(WARD) +
               PRICE:C(AC) + PRICE:ROOMS + PRICE:BEDRM +
               PRICE:C(CNDTN) + PRICE:C(WARD) + C(CNDTN):C(WARD)
"""
model_inter = smf.glm(formula=formula_inter, data=Final_T2,
                      family=sm.families.Binomial()).fit()
print(model_inter.summary())

# Final model (with transformations & interactions)
formula_final = """
QUALIFIED_2 ~ PRICE + np.sqrt(PRICE) + C(AC) + ROOMS + np.power(ROOMS, 0.2) +
               np.sqrt(BEDRM) + C(CNDTN) + C(WARD) +
               PRICE:C(AC) + PRICE:ROOMS + PRICE:C(WARD)
"""
final_model = smf.glm(formula=formula_final, data=Final_T2,
                      family=sm.families.Binomial()).fit()
print(final_model.summary())


# =========================
# 7. Custom ROC Analysis (Training + Validation)
# =========================

def roc_analysis(model, data, response_col, newplot=True, label="Model"):
    """
    Approximate of your R roc.analysis function.
    Returns dict with AUC, cutoff, sensopt, specopt, plus arrays.
    """
    y_true = data[response_col].values
    y_prob = model.predict(data)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    # Lift-like criterion: maximize (tpr - fpr)
    lift = tpr - fpr
    idx = np.argmax(lift)
    cutoff = thresholds[idx]
    sensopt = tpr[idx]
    specopt = 1 - fpr[idx]

    if newplot:
        plt.figure()
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:0.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("1 - Specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "area": auc_val,
        "cutoff": cutoff,
        "sensopt": sensopt,
        "specopt": specopt
    }


trainingROC = roc_analysis(final_model, Final_T2, "QUALIFIED_2", newplot=True, label="Final Model (Train)")
print("Train AUC:", trainingROC["area"])
print("Train cutoff:", trainingROC["cutoff"])
print("Train sensopt:", trainingROC["sensopt"])
print("Train specopt:", trainingROC["specopt"])

validationROC = roc_analysis(final_model, Final_V2, "y", newplot=False, label="Final Model (Valid)")
print("Valid AUC:", validationROC["area"])
print("Valid cutoff:", validationROC["cutoff"])
print("Valid sensopt:", validationROC["sensopt"])
print("Valid specopt:", validationROC["specopt"])


# =========================
# 8. Confusion Matrices & Classification Metrics
# =========================

cutoff = trainingROC["cutoff"]

# Training
train_prob = final_model.predict(Final_T2)
train_pred = (train_prob >= cutoff).astype(int)
train_true = Final_T2["QUALIFIED_2"].values

cm_train = confusion_matrix(train_true, train_pred)
print("Training Confusion Matrix:\n", cm_train)

train_accuracy = accuracy_score(train_true, train_pred)
train_sens = recall_score(train_true, train_pred)  # sensitivity
train_spec = recall_score(train_true, train_pred, pos_label=0)
train_prec = precision_score(train_true, train_pred)
train_f1 = f1_score(train_true, train_pred)

print("Training metrics:")
print("Accuracy:", train_accuracy)
print("Sensitivity:", train_sens)
print("Specificity:", train_spec)
print("Precision:", train_prec)
print("F1:", train_f1)

# Validation
valid_prob = final_model.predict(Final_V2)
valid_pred = (valid_prob >= cutoff).astype(int)
valid_true = Final_V2["y"].values

cm_valid = confusion_matrix(valid_true, valid_pred)
print("Validation Confusion Matrix:\n", cm_valid)

valid_accuracy = accuracy_score(valid_true, valid_pred)
valid_sens = recall_score(valid_true, valid_pred)
valid_spec = recall_score(valid_true, valid_pred, pos_label=0)
valid_prec = precision_score(valid_true, valid_pred)
valid_f1 = f1_score(valid_true, valid_pred)

print("Validation metrics:")
print("Accuracy:", valid_accuracy)
print("Sensitivity:", valid_sens)
print("Specificity:", valid_spec)
print("Precision:", valid_prec)
print("F1:", valid_f1)


# =========================
# 9. Model Comparison Table (AIC, BIC, LogLik, AUC)
# =========================

def extract_metrics(model, name, data, response_col):
    y_true = data[response_col].values
    y_prob = model.predict(data)
    auc_val = roc_auc_score(y_true, y_prob)
    return {
        "Model": name,
        "AIC": model.aic,
        "BIC": model.bic,
        "LogLik": model.llf,
        "AUC": auc_val
    }

model_comparison = pd.DataFrame([
    extract_metrics(basic_model, "Basic Model", Final_T2, "QUALIFIED_2"),
    extract_metrics(model3, "Full Model (No Interactions)", Final_T2, "QUALIFIED_2"),
    extract_metrics(model_aic, "AIC-Selected Model", Final_T2, "QUALIFIED_2"),
    extract_metrics(model_bic, "BIC-Selected Model", Final_T2, "QUALIFIED_2"),
    extract_metrics(final_model, "Final Model", Final_T2, "QUALIFIED_2")
])

print("Model Comparison:\n", model_comparison)


# =========================
# 10. Calibration Plot
# =========================

prob_true, prob_pred = calibration_curve(train_true, train_prob, n_bins=10)

calib_df = pd.DataFrame({
    "mean_pred": prob_pred,
    "mean_obs": prob_true
})

plt.figure()
sns.lineplot(x="mean_pred", y="mean_obs", data=calib_df, marker="o")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Proportion")
plt.title("Calibration Plot — Final Model (Train)")
plt.grid(True)
plt.show()


# =========================
# 11. Lift Chart
# =========================

lift_df = pd.DataFrame({
    "truth": train_true,
    "prob": train_prob
})

lift_df["decile"] = pd.qcut(lift_df["prob"], 10, labels=False) + 1

lift_summary = (
    lift_df
    .groupby("decile")
    .agg(
        avg_prob=("prob", "mean"),
        event_rate=("truth", "mean")
    )
    .reset_index()
)

overall_event_rate = lift_df["truth"].mean()
lift_summary["lift"] = lift_summary["event_rate"] / overall_event_rate

plt.figure()
sns.barplot(x="decile", y="lift", data=lift_summary, color="steelblue")
plt.xlabel("Decile (Predicted Probability)")
plt.ylabel("Lift")
plt.title("Lift Chart — Final Model (Train)")
plt.grid(True, axis="y")
plt.show()


# =========================
# 12. Final Model Performance Table
# =========================

final_model_performance = pd.DataFrame({
    "AIC": [final_model.aic],
    "BIC": [final_model.bic],
    "LogLik": [final_model.llf],
    "Train_AUC": [trainingROC["area"]],
    "Train_Cutoff": [trainingROC["cutoff"]],
    "Train_Sens": [trainingROC["sensopt"]],
    "Train_Spec": [trainingROC["specopt"]],
    "Valid_AUC": [validationROC["area"]],
    "Valid_Cutoff": [validationROC["cutoff"]],
    "Valid_Sens": [validationROC["sensopt"]],
    "Valid_Spec": [validationROC["specopt"]]
})

print("Final Model Performance:\n", final_model_performance)
