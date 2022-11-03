from sklearn.metrics import make_scorer
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, concordance_index_ipcw
import numpy as np
from sksurv.util import Surv
from scipy.stats import pearsonr, spearmanr


def check_target(y):
    is_surv = False
    try:
        b = [x[1] for x in y]
        is_surv = True
    except:
        pass

    if not is_surv:
        return Surv.from_arrays(event=np.full(len(y), True),
                                time=y)
    return y

def harrell_cindex(y_true, y_pred):
    y_true = check_target(y_true)
    return concordance_index_censored(event_indicator=np.array([x[0] for x in y_true]).astype('bool'),
                                      event_time=np.array([x[1] for x in y_true]).astype('float32'),
                                      estimate=y_pred)[0]

def uno_cindex(y_true, y_pred):
    y_true = check_target(y_true)
    event = np.array([x[0] for x in y_true]).astype('bool')
    time = np.array([x[1] for x in y_true]).astype('float32')
    tau = time[event].max()
    return concordance_index_ipcw(y_true, y_true, y_pred, tau)[0]


def tAUC(y_true, y_pred):
    y_true = check_target(y_true)
    evaluation_times_train = np.nanpercentile(np.array([x[1] for x in y_true]).astype('float32'),
                                              np.linspace(10, 81, 25))
    _, auc = cumulative_dynamic_auc(y_true, y_true, y_pred, evaluation_times_train)
    return auc

from sksurv.util import Surv

harrell_cindex_scorer = make_scorer(score_func=harrell_cindex,
                                    greater_is_better=True)

uno_cindex_scorer = make_scorer(score_func=uno_cindex,
                                greater_is_better=True)


tAUC_scorer = make_scorer(score_func=tAUC,
                          greater_is_better=True)

def abs_cindex(x):
    return np.max([x, 1.-x])


def pearson_eval(x,y):
    return pearsonr(x,y)[0]

def spearman_eval(x,y):
    return spearmanr(x,y)[0]

