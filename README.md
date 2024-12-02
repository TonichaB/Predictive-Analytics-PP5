# Property Value Analytics

- Introduction Section
- Project Image

View the live project here (link to be added)

# Contents Table:
- [Property Value Analytics](#property-value-analytics)
- [Contents Table:](#contents-table)
  - [Dataset Content](#dataset-content)
    - [Source](#source)
    - [Licensing](#licensing)
    - [Structure](#structure)
    - [Dataset Quality and Observations](#dataset-quality-and-observations)
  - [Business Requirements](#business-requirements)
  - [Hypothesis and how to validate?](#hypothesis-and-how-to-validate)
    - [Hypotheses](#hypotheses)
    - [Validation Plan](#validation-plan)
  - [The rationale to map the business requirements to the Data Visualisations and ML tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
  - [ML Business Case](#ml-business-case)
  - [User Stories](#user-stories)
  - [Dashboard Design](#dashboard-design)
  - [Methodology](#methodology)
    - [CRISP-DM](#crisp-dm)
    - [Agile Methodology](#agile-methodology)
  - [Rationale for the Model](#rationale-for-the-model)
  - [Project Features](#project-features)
  - [Project Outcomes](#project-outcomes)
  - [Hypothesis Outcomes](#hypothesis-outcomes)
  - [Testing](#testing)
  - [Deployment](#deployment)
    - [Heroku](#heroku)
    - [Fork the Repository](#fork-the-repository)
    - [Cloning the Repository](#cloning-the-repository)
  - [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
  - [Credits](#credits)
    - [Content](#content)
    - [Media](#media)
  - [Acknowledgements](#acknowledgements)

## Dataset Content

### Source
The datasets used in this project were sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data):
1. **House Prices Dataset**: Contains house attribute data (including sale prices) for properties in Ames, Iowa.
2. **Inherited Houses Dataset**: Contains data on four inherited properties provided by our fictitious client, including their attributes but excluding sale prices.

### Licensing

Both datasets are publicly available, and no ethical or privacy concerns are associated with their use.

### Structure

The data is broken down into two datasets:
1. **`house_prices_records.csv`**:
   - 1460 rows and 24 columns.
   - Includes house attributes such as square footage, lot size, and sale prices.
2. **`inherited_houses.csv`**:
   - 4 rows and 23 columns.
   - Shares the same overall structure as `house_prices_records.csv` other than the excluded sale prices column.

<details>
  <summary>Click to view the detailed feature descriptions</summary>

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinished; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

</details>

### Dataset Quality and Observations

The quality of the datasets were inspected within the [Data Collection Notebook](https://github.com/TonichaB/Predictive-Analytics-PP5/blob/main/jupyter_notebooks/01_Data_Collection.ipynb). The key observations made at the time of data collection were as follows;

- **`house_prices_records.csv`**:
  - Contains missing values in columns such as `LotFrontage`, `GarageYrBlt`, and `BedroomAbvGr`.
  - Includes a mixture of data types: integers, floats and objects.
- **`inherited_houses.csv`**:
  - Contains no missing values and is clean.
  - Sale prices are not included, which will require modeling to predict.

The differences in structure and size between the datasets are addressed during the data cleaning and integration steps of the project. Copies of the datasets are saved in the project directory `outputs/datasets/raw` for reproducibility and future reference.

After the data cleaning stage:
- All missing values were handled using appropriate imputation methods.
- Variables with high percentages of missing values (e.g., `WoodDeckSF`) were dropped due to their limited contribution to analysis.
- Duplicate rows and inconsistencies were removed, ensuring data integrity.
- Data types were standardized for seamless processing.

The cleaned datasets are saved in the `outputs/datasets/processed/cleaned/` directory.

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and how to validate?

### Hypotheses

1. **Attribute Correlation**: 
  - Certain house attributes, such as the overall quality (OverallQual), living area size (GrLivArea), and garage area (GarageArea), will have a strong positive correlations with house sale prices in Ames, Iowa.
  - Attributes such as the year of construction (YearBuilt) and basement finish quality (BsmtFinType1) will moderately impact sale prices.
  
2. **Predictive Model**:
  - A regression-based machine learning model trained on the `house_prices_records.csv` dataset will achieve an R2 score of at least 0.75 when tested on unseen data, meeting the client's performance criteria.

3. **Inherited Properties**: 
  - The trained regression model will predict sale prices for the four inherited properties that align with the Ames market trends.

### Validation Plan

1. **Attribute Correlation**:
  - Perform correlation analysis (e.g., Pearson or Spearman) and visualize the results using heatmaps and scatter plots.
  - Validate the strength of correlations statistically and indentify the most impactful features.

2. **Predictive Model**:
  - Develop a regression model using scikit-learn or similar frameworks.
  - Evaluate the model using training and testing datasets, with metrics such as R1 score, mean absolute error (MAE), and root mean squared error (RMSE).

3. **Inherited Properties**:
  - Input the inherited property attributes into the final model and compare the predicted sale prices with historical Ames market data or client expectations.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

## ML Business Case

* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## User Stories

## Dashboard Design

- Page 1
- Page 2
- Page 3
- Page 4
- Page 5

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

## Methodology

### CRISP-DM

This project follows the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology:
1. **Business Understanding**:
  - The client wants to maximize the sale price of four inherited houses in Ames, Iowa.
  - The client requires insights into which attributes correlate with house prices and a predictive model for sale prices.

2. **Data Understanding**:
  - Conducted through the [Data Collection Notebook](https://github.com/TonichaB/Predictive-Analytics-PP5/blob/main/jupyter_notebooks/01_Data_Collection.ipynb):
    - Inspected the dataset structure, quality, and limitations.
    - Summarized the dataset characteristics and initial observations in the notebook and within this README.md file.

3. **Data Preparation**:

This stage involves preparing the raw datasets for analysis and modeling. The key steps are documented in the [Data Cleaning Notebook](https://github.com/TonichaB/Predictive-Analytics-PP5/blob/main/jupyter_notebooks/02_Data_Cleaning.ipynb):

- **Key Objectives**:
  1. Handle missing values in the datasets using imputation techniques.
  2. Remove duplicate rows and address inconsistencies in feature values.
  3. Standardize formatting and ensure compatibility of data types.
  4. Split the cleaned dataset into training and testing sets for modeling.
  5. Save the cleaned and split datasets for future use.

- **Outputs**:
  - Cleaned datasets are saved in `outputs/datasets/processed/cleaned/`.
  - Training and testing datasets are saved in `outputs/datasets/processed/split/`.

1. **Modeling**:

2. **Evaluation**:

3. **Deployment**:

### Agile Methodology

## Rationale for the Model

## Project Features

- **Data Cleaning and Preparation**: The raw datasets were cleaned, formatted and split into training and testing sets. This ensures the data is ready for exploratory analysis and predictive modeling.

## Project Outcomes

- Business Requirement 1
- Business Requirement 2

## Hypothesis Outcomes

1. **Attribute Correlation**:


2. **Predictive Model**:


3. **Inherited Properties**:

## Testing

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

### Fork the Repository

### Cloning the Repository

## Main Data Analysis and Machine Learning Libraries

- Languages (python)
- Frameworks, Libraries & Programs used

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

### Media

## Acknowledgements
