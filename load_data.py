import numpy as np
import xlrd
import hyperparams as hp
hp = hp.create_hparams()

def excel_reader(excel_file, sheet, outcome_col):
    # reads the entire set (specified in hyperparam file) of features in the sheet
    # it assumes that the first 3 columns are patient info and thus are excluded
    wb = xlrd.open_workbook(excel_file)
    sheet = wb.sheet_by_name(sheet)
    num_patients = sheet.nrows - 1
    num_features = hp.num_features
    features = np.zeros([num_patients, num_features])
    names = sheet.col_values(1)[1:]
    keys = sheet.row_values(0)

    for subj in range(num_patients):
        subj_features = sheet.row_values(subj + 1)[3:3 + num_features]
        float_features = list(map(lambda x: np.nan if x == 'None' else x, subj_features))
        features[subj, :] = np.asarray(float_features)
    outcomes = sheet.col_values(keys.index(outcome_col))[1:]

    return names, features, np.asarray(outcomes), keys[3:3+num_features]


def load_data(dataset_path, sheet, outcome_col):
    P, X, Y, Col = excel_reader(dataset_path, sheet, outcome_col)

    return np.abs(X.astype(float)), Y.astype(float), P, Col

