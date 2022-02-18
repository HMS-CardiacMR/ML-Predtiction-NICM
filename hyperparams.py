from collections import namedtuple

HParams = namedtuple(
    "HParams",
    [
        "num_features",
        "use_grid_search",
        "num_pat_excluded",

        "weights_path",
        "results_path",

        "testing_dataset",
        "outcome_col_name",
        "sheet_name",

        "dataset_path_bwh",
        "dataset_path_bi",

    ])


def create_hparams():
    cohort = 'NICM'
    return HParams(
        num_features = 46,
        use_grid_search = True,

        num_pat_excluded = 0, # For manuscript main experiment: set to zero
                                # to test effect of changing event rate, set to 3

        sheet_name= 'Ready_for_Network_46p', # same for BIDMC and BWH sheets
        outcome_col_name='Death+ Cardiac Hosp',
        testing_dataset = 'bidmc', # 'bwh' or 'bidmc'

        weights_path='/models/' + cohort,
        dataset_path_bi='../datasets/BIDMC.xlsx',
        dataset_path_bwh='../datasets/BWH.xlsx',

        results_path='/results/' + cohort
    )
