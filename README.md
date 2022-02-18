# ML-Predtiction-NICM
Code for Fahmy et al. "An Explainable Machine Learning Approach Reveals Prognostic Significance of Right Ventricular Dysfunction in Nonischemic Cardiomyopathy". JACC Cardiovasc Imaging. 2022 doi: 10.1016/j.jcmg.2021.11.029.
## Description of Code
This python code implements our machine learning (ML) model for predicting cardiovascular hospitalization and all-cause death in NICM patients. The input is a vector of clinical variables stored in XLS sheet. The output is the probability of the patient risk of an adverse outcome. Three files are provided:

Hyperparams.py: contains hyperparameters of the algorithm and data (e.g. number of clinical variables, cohort to be tested, etc )
Load_data.py: reads excel sheet and loads the feature vectors (of all patients) into the memory.
Train.py: trains the ML model using the data and also applies the trained model to the selected testing dataset (internal or external).

### Notes:
You may select one patient to display its shap analysis by selecting patient index in line 218 (p = …).
Since we have 10 cross-validations (random splitting of the data), you may display all cross-validation results or select only one of them for display.
The average rank of the clinical variables  (over the 10 cross-validations) is calculated at the end of the file (under section: Summary of CV results).

## Abstract of paper:
Objectives: We implemented an explainable machine learning (ML) model to gain insight into the association between cardiovascular MR (CMR) imaging markers and adverse outcomes of cardiovascular (CV) hospitalization and all-cause death (composite endpoint) in patients with non-ischemic dilated cardiomyopathy (NICM).

Background: Risk stratification of patients with NICM remains challenging. An explainable ML model has the potential to provide insight into the contributions of different risk markers in the prediction model.

Methods: An explainable ML model based on eXtreme gradient boosting machines (XGBoost) was developed using CMR and clinical parameters. The study cohorts consist of NICM patients from two academic medical centers: Beth Israel Deaconess Medical Center (BIDMC) and Brigham and Women’s Hospital (BWH), with 328 and 214 patients, respectively. XGBoost was trained on 70% of patients from the BIDMC cohort and evaluated based on the other 30% as internal validation. The model was externally validated using the BWH cohort. To investigate the contribution of different features in our risk prediction model, we used SHapley Additive exPlanations (SHAP) analysis.

Results: During a mean follow-up duration of over 40 months, 34 patients from BIDMC and 33 patients from BWH experienced the composite endpoint. The area under the curve for predicting the composite endpoint was 0.71 for the internal BIDMC validation and 0.69 for the BWH cohort. SHAP analysis identified parameters associated with right ventricular (RV) dysfunction and remodeling as primary markers of adverse outcomes. High risk thresholds were identified by SHAP analysis and thus provided thresholds for top predictive continuous clinical variables.

Conclusions: An explainable ML-based risk prediction model has the potential to identify NICM patients at risk for CV hospitalization and all-cause death. RV ejection fraction, end-systolic and end-diastolic volumes, as indicators of RV dysfunction and remodeling, were determined as major risk markers.
