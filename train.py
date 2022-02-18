"""
This script trains a XGBM for NICM data.
"""
import hyperparams as hp
from load_data import load_data
import matplotlib.pyplot as plt
from scipy import interp
import pandas as pd
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import shap
init_seed = 2021
num_folds = 10
np.random.seed(init_seed)

shap.initjs()

# 10-fold cross-validation with early stopping
def modelfit(alg, dtrain, predictor, target, useTrainCV=True, cv_folds=num_folds, early_stopping_rounds=25, my_seed = 0):
    np.random.seed(my_seed)

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictor].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain,
                    num_boost_round=alg.get_params()['n_estimators'], # this is the best performing model during CV
                    stratified=True, nfold=cv_folds, verbose_eval=False, metrics='auc', early_stopping_rounds=early_stopping_rounds, seed=my_seed)
        if cvresult.shape[0] > 4:
            alg.set_params(n_estimators=cvresult.shape[0]) # update the model
        else:
            print('too low')
        print("n_estimator: %d" % alg.get_params()['n_estimators'])

    # Fit the best model on the development data (train+validation)
    alg.fit(dtrain[predictor], dtrain[target], eval_metric='auc')

##########################

if __name__ == '__main__':
    # load dataset
    hp = hp.create_hparams()

    sheet_name = hp.sheet_name
    outcome_name = hp.outcome_col_name
    num_features = hp.num_features

    outcome_title = hp.testing_dataset + '_xgbm_CH+death'
    pdirname = os.path.dirname(__file__)

    # Load main dataset for model training and internal validation
    clin_params, outcomes, patients_id, params_names = load_data(dataset_path= hp.dataset_path_bi, # path to training dataset: always BIDMC
                                                                 sheet=sheet_name,
                                                                 outcome_col=outcome_name,
                                                                 normalize=hp.data_norm)
    # load external testing dataset
    if hp.testing_dataset is 'bwh':
        clin_params_ext, outcomes_ext, patients_id_ext, params_names_ext = load_data(dataset_path=hp.dataset_path_bwh,
                                                                     sheet=sheet_name,
                                                                     outcome_col=outcome_name
                                                                     )

    # split the data into train-test(straitified) and give names to columns
    col_names = [str(i + 1) + '_' + params_names[i] for i in range(clin_params.shape[1])]

    ## BIDMC#########################################################################################################################
    rr = np.expand_dims(outcomes, 1) # dummy array used for stratification: = outcome array
    uv,ur,uc = np.unique(feats,return_inverse=True, return_counts=True)

    required_display_seed =-1  # display results at this particular value of seed (i.e. cross-validation);
                                # set to -1: to displays All runs/seeds

    hyperparams_array= []  # store hyperparams at different seeds
    rank_valid_array = []  # store rank of each predictor at different seeds for validation/development
    auc_valid_array  = []  # store auc value at different seeds (for validation)
    auc_test_array   = []  # store auc value at different seeds (for testing)
    tp_test_array    = []  # store ROC curve at different seeds (for testing)
    fp_test_array    = []  # store ROC curve at different seeds (for testing)
    thr_test_array   = []  # store thresholds at different seeds (for testing)
    std_fp   = np.linspace(0, 1, 100) # used to resample FP values

    N = 1 # Set to 10 (10 cross-validation with replacement) to study effect of data split on results
    for idx_seed in range(N): # each seed represents a cross-validation with replacement
        my_seed = init_seed + idx_seed  # Change seed point
        print('################# Starting Seed %d #####################' % my_seed)
        np.random.seed(my_seed)
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(clin_params,outcomes,patients_id, stratify=rr, test_size=0.30, random_state=my_seed)
        dum_idx = np.where(y_train == 1)[0]
        for excl_ind in range(hp.num_pat_excluded):
            X_train = np.delete(X_train, dum_idx[-1-excl_ind], axis=0)
            y_train = np.delete(y_train, dum_idx[-1-excl_ind], axis=0)

        if hp.testing_dataset is 'bwh':
            X_test = clin_params_ext
            y_test = outcomes_ext
            id_test= patients_id_ext

        target_col = outcome_name

        df_train = pd.DataFrame(X_train, columns=col_names)
        df_train.loc[:, target_col] = y_train
        df_test = pd.DataFrame(X_test, columns=col_names)
        df_test.loc[:, target_col] = y_test
        ###################################################################################################
        # Choose all predictors except target & IDcols
        predictors = [x for x in df_train.columns if x not in [target_col]]
        ###################################################################################################
        # stepwise grid search for hyperparameter tuning
        if hp.use_grid_search is True:
            params1 = {'learning_rate': 0.001, 'n_estimators': 500, 'max_depth': 4, 'min_child_weight': 9,
                       'gamma': 0, 'subsample': 0.75, 'colsample_bytree': 0.25, 'objective': 'binary:logistic', 'nthread': 3,
                       'scale_pos_weight': 1, 'seed': my_seed, 'base_score': 0.2,
                       'eval_metric':'logloss', 'use_label_encoder':False #
                       }
            param_test1 = {
                'max_depth': np.arange(3, 10, 2), # high value --> overfit
                'min_child_weight': np.arange(2, 8, 1),# min # points per node in the tree (small --> may overfit)
                'learning_rate': [r for r in np.arange(0.001, 0.01, 0.003)],
                'subsample': [r for r in np.arange(0.5, 1.0, 0.25)], # increase to avoid overf
                'colsample_bytree': [r for r in np.arange(0.25, 1.0, 0.25)] }
            gsearch1 = GridSearchCV(estimator=XGBClassifier(**params1),
                                    param_grid=param_test1, scoring='roc_auc', n_jobs=4, #iid=True, #Remove for sklearn version 0.24.2
                                    cv=num_folds)
            gsearch1.fit(df_train[predictors], df_train[target_col])
            print('Output of grid searh 1: \n', gsearch1.best_params_, gsearch1.best_score_)
            params1.update(gsearch1.best_params_)
            hyperparams_array.append(gsearch1.best_params_)
        else:# hard-code optimal hyper-parameters for first XRS-Valid (indx_seed=0)
            ## Param optimization was obtained using SkLearn version 0.20.2
            params1 = {'learning_rate': 0.001, 'n_estimators': 500, 'max_depth': 3, 'min_child_weight':6,
                   'gamma': 0, 'subsample': 0.5, 'colsample_bytree': 0.75, 'objective': 'binary:logistic',
                   'scale_pos_weight': 1, 'seed': my_seed, 'base_score': 0.2, 'nthread': 3,
                   'eval_metric': 'logloss', 'use_label_encoder': False  #
                   }

        hyperparams_array.append(params1)
        #################################################
        ## Retrain with all development dataset using optimal hyper-parameters
        my_xgb = XGBClassifier(**params1)
        modelfit(my_xgb, df_train, predictors, target_col, early_stopping_rounds=25, my_seed=my_seed)
        final_gbc = my_xgb
        # Predict training and testing datasets:
        dtrain_predictions = final_gbc.predict(df_train[predictors])
        dtrain_predprob = final_gbc.predict_proba(df_train[predictors])[:, 1]
        dtest_predictions = final_gbc.predict(df_test[predictors])
        test_predprob = final_gbc.predict_proba(df_test[predictors])[:, 1]
        ###################################################################################################
        ## Explain model operation based on training dataset
        explainer_shap = shap.TreeExplainer(final_gbc, df_train[predictors])
        shap_values    = explainer_shap.shap_values(df_train[predictors], check_additivity=False)
        predictor_rank = np.sum(np.abs(shap_values), axis=0) # Notice: index startys from ZERO
        r_indx = (np.flip(np.argsort(predictor_rank)))  # index of dominant predictors (descending order)
        print('The predictor list ordered according to importance (Seed %d):'%my_seed)
        tt = ''
        for r in r_indx:
            tt = tt + '   ' + col_names[r]  # index of dominant predictors (descending order)
        print(tt)

        rank_valid_array.append(predictor_rank)
        ###################################################################################################
        fp, tp, thresholds = roc_curve(df_test[target_col], test_predprob)
        auc_ = auc(fp, tp)
        auc_test_array.append(auc_)
        fp_test_array.append(std_fp) # fixed standard fp
        tp_test_array.append(interp(std_fp, fp, tp)) #resample tp at standard FP points: for easy ROC drawing
        thr_test_array.append(thresholds)
        #####################################################################################################
        ## You may want to display results for a particular run or for all runs
        if idx_seed == required_display_seed or required_display_seed == -1:
            print('## Results at Seed %d ##' % my_seed)
            print("Accuracy (train): %.4g" % accuracy_score(df_train[target_col].values, dtrain_predictions))
            print("AUC (Train): %f" % roc_auc_score(df_train[target_col], dtrain_predprob))
            print("Accuracy (Test): %.4g" % accuracy_score(df_test[target_col].values, dtest_predictions))
            print("AUC (Test): %f" % roc_auc_score(df_test[target_col], test_predprob))

            plt.figure(idx_seed*100+1 , figsize=(4, 4))
            dummy = shap_values
            dummy = np.abs(np.sqrt(np.abs(dummy))) # contrast stretching for better visualization
            dummy = np.multiply(dummy, np.sign(shap_values))
            shap.summary_plot(dummy, df_train[predictors], plot_type='dot', max_display=12, show=False, plot_size=(10, 6))
            plt.savefig(pdirname + './results/NICM-SHAPdot_XGBM' + outcome_title + "_Seed_%d.png" %idx_seed, dpi=300)

            plt.figure(idx_seed*100+2 )
            shap.summary_plot(shap_values, df_train[predictors], plot_type='bar',max_display=12, show=False, plot_size=(10,6))
            plt.savefig(pdirname + './results/NICM-SHAPbar_XGBM' + outcome_title + "_Seed_%d.png" %idx_seed, dpi=300)

            plt.figure(idx_seed*100+3, figsize=(4, 4))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(std_fp, tp_test_array[idx_seed], lw=3, color='b', label='AUC = %0.3f' % auc_)
            plt.xlim([-0.05, 1.0])
            plt.ylim([-0.05, 1.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right", prop={'size': 12})
            plt.savefig(pdirname + './results/NICM-ROC_XGBM' + outcome_title + "_Seed_%d.png" %idx_seed, dpi=300)

            plt.figure(idx_seed*100+4, figsize=(4, 4))
            plt.plot([0, 1], [0.1, 0.1], 'k--')
            pr,rec,th = precision_recall_curve(df_test[target_col],test_predprob)
            AP = average_precision_score(df_test[target_col],test_predprob)
            plt.plot(rec, pr, lw=3, color='b', label='AP = %0.3f' % AP)
            plt.xlim([-0.0, 1.0])
            plt.ylim([-0.0, 1.01])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right", prop={'size': 12})
            plt.savefig(pdirname + './results/NICM-RecPre_XGBM' + outcome_title + "_Seed_%d.png" %idx_seed, dpi=300)
            print(np.where(y_train==1))
            print('#####################')

            dum = df_train[predictors]
            p = 45 # Select one patient at random for display
            print('Patient ID in XLS Sheet = ', id_train[p])
            shap.force_plot(explainer_shap.expected_value, shap_values[p, :],
                            np.round(dum.iloc[p, :]), feature_names= col_names,
                            text_rotation=25, matplotlib=True,show=False)
            print('Ground truth     =  ', y_train[p])
            print('Prediction Prob  =  ', dtrain_predprob[p])
            print('Base value  =  ', explainer_shap.expected_value)
            print('SHAP value', np.sum(shap_values[p, :]) + explainer_shap.expected_value)
            print('#####################')
            plt.savefig(pdirname + './results/NICM-SHAPforce1_XGBM' + outcome_title + "_Seed_%d.png" %idx_seed, dpi=300)

            # Figure 4 in manuscript; show results of the first (main) run (or seed)
            if idx_seed ==0:
                for i in range(len(predictors)):
                    shap.dependence_plot(i, shap_values, df_train[predictors],
                    interaction_index = None, show = False)
                    plt.ylabel('Shap values for' + str(params_names[i]))
                    plt.xlabel(str(params_names[i]))
                    dumstr = str(params_names[i]).replace(" ", "")
                    dumstr = dumstr.replace("(","")
                    dumstr = dumstr.replace(")", "")
                    dumstr = dumstr.replace("/", "")
                    dumstr = dumstr.replace("?", "")
                    plt.savefig(pdirname + './results/NICM_SHAP_' + dumstr + '_XGBM' + outcome_title + ".png", dpi=300)
                    plt.close()

            final_gbc.save_model(
                pdirname + './models/NICM-XGBM-%d' % num_features + 'Param-' + outcome_title+
                                '_Seed_%d_' %my_seed + 'AUC = %0.3f' % auc_)

        print('Done of seed {seed:d}.'.format(seed=my_seed))

    ## Summary of CV results
    print('######################################')
    print('Summary of Cross-validation Results:')
    print('######################################')
    TheRank = np.mean(np.array(rank_valid_array),0)
    r_indx = np.flip(np.argsort(TheRank))
    print('The predictor list ordered according to importance:')
    tt=''
    for r in r_indx:
        tt = tt+ '   ' +col_names[r]  # index of dominant predictors (descending order)
    print(tt)
    print('Test AUC Average={av: .2f}; Min={Min: .2f};  Max={Max: .2f}'.format(av=np.mean(auc_test_array),Min=np.min(auc_test_array),Max=np.max(auc_test_array)))
    plt.show()
