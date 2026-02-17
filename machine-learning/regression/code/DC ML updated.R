library(tidyverse)
library(readxl)
library(broom)
library(modelr)
library(boot)
library(scales)
library(rms)
library(ggplot2)
library(yardstick)

# LAST EDITTED: February 2, 2026

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
  sample_frac(0.75)

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

# Last Edit and more coming February 17, 2026
