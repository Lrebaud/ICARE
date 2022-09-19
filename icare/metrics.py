from sklearn.metrics import make_scorer
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, concordance_index_ipcw
import numpy as np
from sksurv.util import Surv
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
from sksurv.linear_model import CoxPHSurvivalAnalysis

def hazard_ratio(y_true, y_pred):
    y_pred = (y_pred - np.mean(y_pred)) / np.max([1e-6, np.std(y_pred)])
    event = np.array([x[0] for x in y_true]).astype('bool')
    time = np.array([x[1] for x in y_true]).astype('float32')
    time = (time - np.mean(time)) / np.max([1e-6, np.std(time)])
    y_true = Surv.from_arrays(event=event, time=time)
    cph = CoxPHSurvivalAnalysis()
    cph.fit(y_pred.reshape(-1,1), y_true)
    coef = cph.coef_[0]
    return coef

# def HR()


harrell_cindex_scorer = make_scorer(score_func=harrell_cindex,
                                    greater_is_better=True)

uno_cindex_scorer = make_scorer(score_func=uno_cindex,
                                greater_is_better=True)


tAUC_scorer = make_scorer(score_func=tAUC,
                          greater_is_better=True)


