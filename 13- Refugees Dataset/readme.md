# Refugees Dataset (2010 - 2022) Exploration

## Dataset Overview
The dataset [Refugees](https://www.kaggle.com/datasets/sujaykapadnis/refugees) contains information about refugees, asylum seekers, internally displaced persons (IDPs), and other related categories from 2010 to 2022. The main features include the year, country of origin (COO), country of asylum (COA), number of refugees, asylum seekers, returned refugees, IDPs, returned IDPs, stateless persons, others of concern to UNHCR (OOC), and other people in need of international protection (OIP).

## Data Cleaning
- Dropped columns with excessive NaN values (`oip` and `hst`).
- Removed columns `coo_iso` and `coa_iso` for simplicity.

## Exploratory Data Analysis (EDA)
- Explored the distribution of refugees over the years.
- Analyzed the correlation between different features.
- Investigated the relationship between refugees and IDPs.
- Explored trends and relationships through various visualizations.

## Key Findings
1. The dataset contains information on 64,809 entries with a mix of data types.
2. The top countries with the highest number of refugees include Syria, Afghanistan, and South Sudan.
3. Asylum seekers are notably high in Germany, Turkey, and the United States.
4. The distribution of IDPs shows an overall increasing trend over the years.
5. The refugee to asylum seeker ratio was calculated, with an average ratio indicating varying proportions.

## Recommendations
1. Further analysis could focus on specific regions or continents to provide more targeted insights.
2. Considering the increasing trend in IDPs, exploring the causes and regions affected would be valuable.

## Sources
1. [Refugees Dataset on Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/refugees)