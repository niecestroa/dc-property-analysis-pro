library(tidyverse)
library(readxl)
library(broom)
library(modelr)
library(boot)
library(scales)
library(rms)
library(ggplot2)
library(yardstick)

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

DC_Properties <- read_excel(
  "~/Documents/STAT 616 Generalizd Linear Models/GLM Project/Data/DC_Properties.xlsx",
  na = ""
)

# ---------------------------------------------------------
# Visualization Dataset (Cleaned)
# ---------------------------------------------------------

dcproperty <- DC_Properties %>%
  select(
    -CMPLX_NUM, -LIVING_GBA, -SALE_NUM, -GIS_LAST_MOD_DTTM
  ) %>%
  filter(
    PRICE > 10000, PRICE < 10000000,
    HEAT != "No Data",
    CNDTN != "No Data", CNDTN != "Default",
    STRUCT != "Default",
    GRADE != " No Data",
    STYLE != "Default",
    KITCHENS <= 10,
    ROOMS < 26,
    BEDRM < 20,
    STORIES < 100,
    BATHRM > 0,
    HF_BATHRM > 0
  ) %>%
  mutate(
    QUALIFIED_2 = if_else(QUALIFIED == "Q", 1, 0),
    AC = if_else(AC == "0", "N", AC),
    GRADE = case_when(
      GRADE %in% c("Exceptional-A", "Exceptional-B", "Exceptional-C", "Exceptional-D") ~ "Exceptional",
      TRUE ~ GRADE
    )
  ) %>%
  drop_na()

# ---------------------------------------------------------
# Final Cleaned Dataset
# ---------------------------------------------------------

DC_Final <- DC_Properties %>%
  select(
    -NUM_UNITS, -YR_RMDL, -SALEDATE, -GBA, -STRUCT, -EXTWALL, -ROOF,
    -INTWALL, -CMPLX_NUM, -LIVING_GBA, -FULLADDRESS, -CITY, -STATE,
    -NATIONALGRID, -ASSESSMENT_SUBNBHD, -CENSUS_BLOCK, -SALE_NUM,
    -GIS_LAST_MOD_DTTM
  ) %>%
  filter(
    CNDTN != "No Data", CNDTN != "Default",
    GRADE != " No Data",
    STYLE != "Default",
    PRICE > 10000, PRICE < 10000000,
    FIREPLACES < 8,
    KITCHENS <= 10,
    ROOMS < 26,
    BEDRM < 20,
    STORIES < 100,
    BATHRM > 0,
    HF_BATHRM > 0
  ) %>%
  mutate(
    QUALIFIED_2 = if_else(QUALIFIED == "Q", 1, 0),
    AC = if_else(AC == "0", "N", AC)
  ) %>%
  drop_na()

# ---------------------------------------------------------
# Truly Random Training / Validation Split
# ---------------------------------------------------------

set.seed(123)

Final_T <- DC_Final %>%
  mutate(.id = row_number()) %>%
  sample_frac(0.80)

Final_V <- DC_Final %>%
  anti_join(Final_T, by = ".id") %>%
  select(-.id)

Final_T <- Final_T %>% select(-.id)

# ---------------------------------------------------------
# Model Selection (Upgraded to Tidyverse Pipelines)
# ---------------------------------------------------------

# Convert categorical variables once, cleanly
Final_T2 <- Final_T %>%
  mutate(
    AC      = factor(AC),
    STYLE   = factor(STYLE),
    CNDTN   = factor(CNDTN),
    WARD    = factor(WARD),
    QUALIFIED_2 = factor(QUALIFIED_2)
  )

# ---------------------------------------------------------
# Basic Model
# ---------------------------------------------------------

basic_model <- Final_T2 %>%
  glm(QUALIFIED_2 ~ PRICE,
      data = .,
      family = binomial(link = "logit"))

summary(basic_model)

# ---------------------------------------------------------
# Full Model (No Interactions)
# ---------------------------------------------------------

model3 <- Final_T2 %>%
  glm(
    QUALIFIED_2 ~ PRICE + BATHRM + HF_BATHRM + AC +
      ROOMS + BEDRM + STORIES + STYLE + CNDTN +
      KITCHENS + FIREPLACES + WARD,
    data = .,
    family = binomial(link = "logit")
  )

summary(model3)

# ---------------------------------------------------------
# AIC Model Selection
# ---------------------------------------------------------

model_aic <- model3 %>%
  step(direction = "both") %>%
  glm(
    formula = .,
    data = Final_T2,
    family = binomial(link = "logit")
  )

summary(model_aic)

# ---------------------------------------------------------
# BIC Model Selection
# ---------------------------------------------------------

sampsize <- length(model3$fitted.values)

model_bic <- model3 %>%
  step(direction = "both", k = log(sampsize)) %>%
  glm(
    formula = .,
    data = Final_T2,
    family = binomial(link = "logit")
  )

summary(model_bic)

# ---------------------------------------------------------
# Interaction Model (AIC & BIC Exploration)
# ---------------------------------------------------------

model_inter <- Final_T2 %>%
  glm(
    QUALIFIED_2 ~ PRICE + AC + ROOMS + BEDRM + CNDTN + WARD +
      PRICE:AC + PRICE:ROOMS + PRICE:BEDRM +
      PRICE:CNDTN + PRICE:WARD + CNDTN:WARD,
    data = .,
    family = binomial(link = "logit")
  )

# AIC step
step(model_inter, direction = "both")

# BIC step
sampsize <- length(model_inter$fitted.values)
step(model_inter, direction = "both", k = log(sampsize))

# ---------------------------------------------------------
# Final Model (Your Chosen Specification)
# ---------------------------------------------------------

final_model <- Final_T2 %>%
  glm(
    QUALIFIED_2 ~ PRICE + I(PRICE^0.5) + AC +
      ROOMS + I(ROOMS^.2) + I(BEDRM^0.5) +
      CNDTN + WARD +
      PRICE:AC + PRICE:ROOMS + PRICE:WARD,
    data = .,
    family = binomial(link = "logit")
  )

summary(final_model)

# ---------------------------------------------------------
# Section 2: Visualisation (Upgraded to Tidyverse Pipelines)
# ---------------------------------------------------------

# Ensure factors are set once
Final_T2 <- Final_T %>%
  mutate(
    AC      = factor(AC),
    STYLE   = factor(STYLE),
    CNDTN   = factor(CNDTN),
    WARD    = factor(WARD),
    QUALIFIED_2 = factor(QUALIFIED_2)
  )

# ---------------------------------------------------------
# Diagnostic Plots
# ---------------------------------------------------------

# Extract diagnostics using pipeline
final_model_diag <- final_model %>%
  glm.diag()

# View first 10 of each diagnostic metric
1:10 %>% { final_model_diag$rd[.] }      # Standardized Deviance Residuals
1:10 %>% { final_model_diag$rp[.] }      # Standardized Pearson Residuals
1:10 %>% { final_model_diag$cook[.] }    # Cook's Distance
1:10 %>% { final_model_diag$h[.] }       # Leverage

# Base diagnostic plots
glm.diag.plots(final_model)
plot(final_model)

# ---------------------------------------------------------
# ROC Curve Function (kept identical to your original)
# ---------------------------------------------------------

roc.analysis <- function(object, newdata = NULL, newplot = TRUE) {

  if (is.null(newdata)) {
    pi.tp <- object$fitted[object$y == 1]
    pi.tn <- object$fitted[object$y == 0]
  } else {
    preds <- predict(object, newdata, type = "response")
    pi.tp <- preds[newdata$y == 1]
    pi.tn <- preds[newdata$y == 0]
  }

  pi.all <- sort(c(pi.tp, pi.tn))
  sens <- rep(1, length(pi.all) + 1)
  specc <- rep(1, length(pi.all) + 1)

  for (i in seq_along(pi.all)) {
    sens[i + 1] <- mean(pi.tp >= pi.all[i], na.rm = TRUE)
    specc[i + 1] <- mean(pi.tn >= pi.all[i], na.rm = TRUE)
  }

  npoints <- length(sens)
  area <- sum(0.5 * (sens[-1] + sens[-npoints]) *
                (specc[-npoints] - specc[-1]))

  lift <- (sens - specc)[-1]
  cutoff <- pi.all[lift == max(lift)][1]
  sensopt <- sens[-1][lift == max(lift)][1]
  specopt <- 1 - specc[-1][lift == max(lift)][1]

  if (newplot) {
    plot(specc, sens,
         xlim = c(0, 1), ylim = c(0, 1),
         type = "s",
         xlab = "1 - Specificity",
         ylab = "Sensitivity",
         main = "ROC Curve")
    abline(0, 1)
  } else {
    lines(specc, sens, type = "s", lty = 2, col = 2)
  }

  list(
    pihat = as.vector(pi.all),
    sens = as.vector(sens[-1]),
    spec = as.vector(1 - specc[-1]),
    area = area,
    cutoff = cutoff,
    sensopt = sensopt,
    specopt = specopt
  )
}

# ---------------------------------------------------------
# Training ROC (Pipelined)
# ---------------------------------------------------------

trainingROC <- final_model %>%
  roc.analysis()

trainingROC$area
trainingROC$cutoff
trainingROC$sensopt
trainingROC$specopt

# ---------------------------------------------------------
# Validation ROC (Pipelined)
# ---------------------------------------------------------

Final_V2 <- Final_V %>%
  mutate(y = QUALIFIED_2)

validationROC <- final_model %>%
  roc.analysis(newdata = Final_V2, newplot = FALSE)

validationROC$area
validationROC$cutoff
validationROC$sensopt
validationROC$specopt

# ---------------------------------------------------------
# Section 2: Visualisation (Upgraded to Tidyverse Pipelines)
# ---------------------------------------------------------

# Ensure categorical variables are set once
Final_T2 <- Final_T %>%
  mutate(
    AC      = factor(AC),
    STYLE   = factor(STYLE),
    CNDTN   = factor(CNDTN),
    WARD    = factor(WARD),
    QUALIFIED_2 = factor(QUALIFIED_2)
  )

# ---------------------------------------------------------
# Variance Inflation Factors (VIF)
# ---------------------------------------------------------

vif_model <- Final_T2 %>%
  glm(
    QUALIFIED_2 ~ PRICE + I(PRICE^0.5) + AC +
      ROOMS + I(ROOMS^.2) + I(BEDRM^0.5) +
      CNDTN + WARD +
      PRICE:AC + PRICE:ROOMS + PRICE:WARD,
    data = .,
    family = binomial(link = "logit")
  )

vif(vif_model)

# ---------------------------------------------------------
# Basic Model Graph
# ---------------------------------------------------------

graph <- dcproperty %>%
  ggplot(aes(x = QUALIFIED_2, y = PRICE)) +
  stat_sum() +
  stat_smooth(
    method = "glm",
    method.args = list(family = "binomial"),
    se = TRUE,
    fullrange = TRUE
  ) +
  labs(
    title = "Market Qualification based on Price and Location",
    subtitle = "Ward 2 and Ward 3 are the more expensive places to live",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Qualification to be on Market",
    color = "Qualification"
  ) +
  facet_wrap(~ WARD) +
  theme_bw()

graph

# ---------------------------------------------------------
# Map Graph 1 — Zipcode
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(x = X, y = Y, color = ZIPCODE)) +
  geom_point() +
  labs(
    title = "Map of Data by Zipcode",
    subtitle = "Locations are well distributed",
    caption = "Data from Kaggle.com",
    x = "Latitude",
    y = "Longitude",
    color = "Zipcode"
  ) +
  theme_bw() +
  theme(legend.position = "right")

# ---------------------------------------------------------
# Map Graph 2 — Ward
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(x = X, y = Y, color = WARD)) +
  geom_point() +
  labs(
    title = "Map of Data by Ward",
    subtitle = "Originally there were 8 Wards",
    caption = "Data from Kaggle.com",
    x = "Latitude",
    y = "Longitude",
    color = "Ward"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# Map Graph 3 — Quadrant
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(x = X, y = Y, color = QUADRANT)) +
  geom_point() +
  labs(
    title = "Map of Data by Quadrants",
    subtitle = "Little to no South-West area",
    caption = "Data from Kaggle.com",
    x = "Latitude",
    y = "Longitude",
    color = "Quadrant"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# Year Graph 1 — Quadrant
# ---------------------------------------------------------

dcproperty %>%
  filter(YR_RMDL > 1800) %>%
  ggplot(aes(x = QUADRANT, y = YR_RMDL, color = QUALIFIED)) +
  geom_boxplot() +
  labs(
    title = "Price based on Qualifications and Quadrant",
    subtitle = "Ward 1 to 3 are the most expensive",
    caption = "Data from Kaggle.com",
    x = "Quadrant",
    y = "Year Last Remodeled",
    color = "Qualification"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# Year Graph 2 — Grade
# ---------------------------------------------------------

dcproperty %>%
  filter(YR_RMDL > 1800) %>%
  ggplot(aes(x = GRADE, y = YR_RMDL, color = QUALIFIED)) +
  geom_boxplot() +
  coord_flip() +
  labs(
    title = "Price based on Qualifications and Grade",
    subtitle = "Ward 1 to 3 are the most expensive",
    caption = "Data from Kaggle.com",
    x = "Grade",
    y = "Year Last Remodeled",
    color = "Qualification"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# Year Graph 3 — Condition
# ---------------------------------------------------------

dcproperty %>%
  filter(YR_RMDL > 1800) %>%
  ggplot(aes(x = CNDTN, y = YR_RMDL, color = QUALIFIED)) +
  geom_boxplot() +
  coord_flip() +
  labs(
    title = "Price based on Qualifications and Condition",
    subtitle = "Ward 1 to 3 are the most expensive",
    caption = "Data from Kaggle.com",
    x = "Condition Review",
    y = "Year Last Remodeled",
    color = "Qualification"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

  # ---------------------------------------------------------
# Heat & AC Graph
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(x = AC, y = PRICE)) +
  geom_point(aes(color = HEAT)) +
  facet_wrap(~ QUALIFIED_2) +
  labs(
    title = "Price, AC and Heat",
    subtitle = "AC is Important to the Price",
    caption = "Data from Kaggle.com",
    y = "Price",
    x = "AC",
    color = "Heat"
  ) +
  theme_bw() +
  theme(legend.position = "right")

# ---------------------------------------------------------
# Continuous Variable Plots
# ---------------------------------------------------------

# ROOMS vs PRICE
text_df <- tibble(text = " \n After 18 rooms \n Price decreases", x = -Inf, y = Inf)

dcproperty %>%
  ggplot(aes(ROOMS, PRICE)) +
  geom_point(aes(color = factor(QUALIFIED_2, labels = c("Not Qualified", "Qualified")))) +
  geom_smooth(se = FALSE, color = "black") +
  geom_text(aes(x, y, label = text), data = text_df, vjust = "top", hjust = "left") +
  labs(
    title = "Price Increases as Rooms Increase",
    subtitle = "Most Qualified Properties Are Under $1M",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Number of Rooms",
    color = "Qualification"
  ) +
  scale_colour_brewer(palette = "Set1") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# KITCHENS vs PRICE
# ---------------------------------------------------------

text_df <- tibble(text = "More Kitchens equals\nLower Price & Less Qualified", x = Inf, y = Inf)

dcproperty %>%
  ggplot(aes(KITCHENS, PRICE)) +
  geom_point(aes(color = factor(QUALIFIED_2, labels = c("Not Qualified", "Qualified")))) +
  geom_text(aes(x, y, label = text), data = text_df, vjust = "top", hjust = "right") +
  labs(
    title = "Kitchens Impact on Price",
    subtitle = "Having Fewer Kitchens Increases the Price",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Number of Kitchens",
    color = "Qualification"
  ) +
  scale_colour_brewer(palette = "Set1") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# NUM_UNITS vs PRICE
# ---------------------------------------------------------

text_df <- tibble(text = "After Bedrooms equals 9\nPrice decreases", x = Inf, y = Inf)

dcproperty %>%
  ggplot(aes(NUM_UNITS, PRICE)) +
  geom_point(aes(color = as.factor(QUADRANT))) +
  geom_text(aes(x, y, label = text), data = text_df, vjust = "top", hjust = "right") +
  labs(
    title = "Bedrooms Impact on Price",
    subtitle = "Having too much is a bad thing",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Number of Available Units",
    color = "Quadrant"
  ) +
  scale_colour_brewer(palette = "Set1") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# BEDROOMS vs PRICE
# ---------------------------------------------------------

text_df <- tibble(text = "More Bedrooms equals\nLower Price & Less Qualified", x = -Inf, y = Inf)

dcproperty %>%
  ggplot(aes(BEDRM, PRICE)) +
  geom_point(aes(color = factor(QUALIFIED_2, labels = c("Not Qualified", "Qualified")))) +
  geom_smooth(se = FALSE, color = "black") +
  geom_text(aes(x, y, label = text), data = text_df, vjust = "top", hjust = "left") +
  labs(
    title = "Bedrooms Steep Increase & Decrease on Price",
    subtitle = "Price rises until ~9 bedrooms, then falls",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Number of Bedrooms",
    color = "Qualification"
  ) +
  scale_colour_brewer(palette = "Paired") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# STORIES vs PRICE
# ---------------------------------------------------------

text_df <- tibble(text = "Highest Price is when \n the Number of Stories \n is between 2 to 3", x = Inf, y = Inf)

dcproperty %>%
  ggplot(aes(STORIES, PRICE)) +
  geom_point(aes(color = factor(QUALIFIED_2, labels = c("Not Qualified", "Qualified")))) +
  geom_smooth(se = FALSE, color = "black") +
  geom_text(aes(x, y, label = text), data = text_df, vjust = "top", hjust = "right") +
  labs(
    title = "Stories Increase & Dramatic Decrease with Price",
    subtitle = "Having between 1 and 5 Stories = Higher Price",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Number of Stories",
    color = "Qualification"
  ) +
  scale_colour_brewer(palette = "Paired") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# FIREPLACES vs PRICE
# ---------------------------------------------------------

text_df <- tibble(text = "It seems that the number of Fireplaces\nhas little effect on price", x = Inf, y = -Inf)

dcproperty %>%
  ggplot(aes(FIREPLACES, PRICE)) +
  geom_point(aes(color = factor(QUALIFIED_2, labels = c("Not Qualified", "Qualified")))) +
  geom_smooth(se = FALSE, color = "blue") +
  geom_text(aes(x, y, label = text), data = text_df, vjust = "bottom", hjust = "right") +
  labs(
    title = "Fireplaces and Price",
    subtitle = "Qualification Rate is unchanged as Fireplaces increase",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Number of Fireplaces",
    color = "Qualification"
  ) +
  scale_colour_brewer(palette = "Reds") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# Time Graph — SALEDATE vs PRICE
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(SALEDATE, PRICE)) +
  geom_point(aes(color = as.factor(QUALIFIED_2))) +
  geom_smooth(se = FALSE, color = "red") +
  labs(
    title = "Price Increases as Time Increases",
    subtitle = "More recent sales have higher prices",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Year Last Sold",
    color = "Qualification"
  ) +
  scale_colour_brewer(palette = "BuPu") +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# SALEDATE × PRICE × WARD faceted by GRADE
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(SALEDATE, PRICE)) +
  geom_point(aes(color = factor(WARD))) +
  geom_smooth(se = FALSE, color = "black") +
  labs(
    title = "Stacking of Ward Areas over Years by Grade",
    subtitle = "Lower ward numbers tend to have higher prices",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Year Last Sold",
    color = "Ward"
  ) +
  scale_colour_brewer(palette = "YlOrBr") +
  facet_wrap(~ GRADE) +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# SALEDATE × PRICE × QUADRANT faceted by GRADE
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(SALEDATE, PRICE)) +
  geom_point(aes(color = factor(QUADRANT))) +
  geom_smooth(se = FALSE, color = "black") +
  labs(
    title = "Overlap of Quadrant Price over Years by Grade",
    subtitle = "Northwest consistently sells at the highest price",
    caption = "Data from Kaggle.com",
    y = "Price ($)",
    x = "Year Last Sold",
    color = "Quadrant"
  ) +
  scale_colour_brewer(palette = "YlGnBu") +
  facet_wrap(~ GRADE) +
  theme_bw() +
  theme(legend.position = "bottom")

# ---------------------------------------------------------
# SALEDATE × ROOMS × GRADE faceted by WARD
# ---------------------------------------------------------

dcproperty %>%
  ggplot(aes(SALEDATE, ROOMS)) +
  geom_point(aes(color = factor(GRADE))) +
  geom_smooth(se = FALSE, color = "black") +
  labs(
    title = "Rooms Over Time by Ward",
    subtitle = "Higher‑numbered wards show different room trends over time",
    caption = "Data from Kaggle.com",
    y = "Rooms",
    x = "Year Last Sold",
    color = "Grade"
  ) +
  scale_colour_brewer(palette = "Paired") +
  facet_wrap(~ WARD) +
  theme_bw() +
  theme(legend.position = "right")

# ---------------------------------------------------------
# Extract model fit statistics (AIC, BIC, logLik)
# ---------------------------------------------------------

fit_stats <- tibble(
  AIC  = AIC(final_model),
  BIC  = BIC(final_model),
  LogLik = logLik(final_model)[1]
)

# ---------------------------------------------------------
# Training ROC metrics (using your custom roc.analysis)
# ---------------------------------------------------------

train_roc <- final_model %>%
  roc.analysis() %>%
  { tibble(
      Train_AUC     = .$area,
      Train_Cutoff  = .$cutoff,
      Train_Sens    = .$sensopt,
      Train_Spec    = .$specopt
    )
  }

# ---------------------------------------------------------
# Validation ROC metrics
# ---------------------------------------------------------

Final_V2 <- Final_V %>%
  mutate(y = QUALIFIED_2)

valid_roc <- final_model %>%
  roc.analysis(newdata = Final_V2, newplot = FALSE) %>%
  { tibble(
      Valid_AUC     = .$area,
      Valid_Cutoff  = .$cutoff,
      Valid_Sens    = .$sensopt,
      Valid_Spec    = .$specopt
    )
  }

# ---------------------------------------------------------
# Combine into a single performance table
# ---------------------------------------------------------

final_model_performance <- bind_cols(fit_stats, train_roc, valid_roc)

final_model_performance

# ---------------------------------------------------------
# Helper: Extract model metrics
# Final Tables
# ---------------------------------------------------------

extract_metrics <- function(model, name) {
  tibble(
    Model = name,
    AIC = AIC(model),
    BIC = BIC(model),
    LogLik = logLik(model)[1]
  )
}

# ---------------------------------------------------------
# MODEL COMPARISON TABLE
# ---------------------------------------------------------

model_comparison <- bind_rows(
  extract_metrics(basic_model, "Basic Model"),
  extract_metrics(model3, "Full Model (No Interactions)"),
  extract_metrics(model_aic, "AIC-Selected Model"),
  extract_metrics(model_bic, "BIC-Selected Model"),
  extract_metrics(final_model, "Final Model")
)

model_comparison

# ---------------------------------------------------------
# CONFUSION MATRICES (TRAINING + VALIDATION)
# ---------------------------------------------------------

cutoff <- trainingROC$cutoff

# Training predictions
train_predictions <- final_model %>%
  predict(Final_T2, type = "response") %>%
  { if_else(. >= cutoff, 1, 0) }

train_actual <- Final_T2$QUALIFIED_2 %>% as.numeric() - 1

training_confusion <- tibble(
  Actual = train_actual,
  Predicted = train_predictions
) %>%
  count(Actual, Predicted) %>%
  pivot_wider(names_from = Predicted, values_from = n, values_fill = 0)

training_confusion

# Validation predictions
valid_predictions <- final_model %>%
  predict(Final_V2, type = "response") %>%
  { if_else(. >= cutoff, 1, 0) }

valid_actual <- Final_V2$y %>% as.numeric() - 1

validation_confusion <- tibble(
  Actual = valid_actual,
  Predicted = valid_predictions
) %>%
  count(Actual, Predicted) %>%
  pivot_wider(names_from = Predicted, values_from = n, values_fill = 0)

validation_confusion

# ---------------------------------------------------------
# ACCURACY, SENSITIVITY, SPECIFICITY, PRECISION, F1
# ---------------------------------------------------------

# Training metrics
train_metrics <- tibble(
  truth = factor(train_actual, levels = c(0,1)),
  estimate = factor(train_predictions, levels = c(0,1))
) %>%
  yardstick::metrics(truth, estimate)

train_metrics

# Validation metrics
valid_metrics <- tibble(
  truth = factor(valid_actual, levels = c(0,1)),
  estimate = factor(valid_predictions, levels = c(0,1))
) %>%
  yardstick::metrics(truth, estimate)

valid_metrics

# ---------------------------------------------------------
# AUC COMPARISON TABLE ACROSS ALL MODELS
# ---------------------------------------------------------

compute_auc <- function(model, name) {
  preds <- predict(model, Final_T2, type = "response")
  truth <- Final_T2$QUALIFIED_2 %>% as.numeric() - 1
  
  tibble(
    Model = name,
    AUC = yardstick::roc_auc_vec(
      truth = factor(truth, levels = c(0,1)),
      estimate = preds
    )
  )
}

auc_comparison <- bind_rows(
  compute_auc(basic_model, "Basic Model"),
  compute_auc(model3, "Full Model (No Interactions)"),
  compute_auc(model_aic, "AIC-Selected Model"),
  compute_auc(model_bic, "BIC-Selected Model"),
  compute_auc(final_model, "Final Model")
)

auc_comparison

# ---------------------------------------------------------
# CALIBRATION PLOT
# ---------------------------------------------------------

calibration_df <- tibble(
  truth = factor(train_actual, levels = c(0,1)),
  prob = predict(final_model, Final_T2, type = "response")
)

calibration_df %>%
  mutate(bin = ntile(prob, 10)) %>%
  group_by(bin) %>%
  summarise(
    mean_prob = mean(prob),
    observed = mean(as.numeric(truth) - 1)
  ) %>%
  ggplot(aes(mean_prob, observed)) +
  geom_point(size = 3, color = "blue") +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "Calibration Plot",
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  theme_bw()

# ---------------------------------------------------------
# LIFT CHART
# ---------------------------------------------------------

lift_df <- tibble(
  truth = train_actual,
  prob = predict(final_model, Final_T2, type = "response")
) %>%
  mutate(bin = ntile(prob, 10)) %>%
  group_by(bin) %>%
  summarise(
    avg_prob = mean(prob),
    lift = mean(truth) / mean(train_actual)
  )

lift_df %>%
  ggplot(aes(bin, lift)) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Lift Chart",
    x = "Decile (Predicted Probability)",
    y = "Lift"
  ) +
  theme_bw()
