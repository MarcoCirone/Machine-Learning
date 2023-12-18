import numpy as np
from models.logistic_regression.log_reg_training_models import LR
from general.utils import compute_min_dcf
from general.evaluation import get_evaluation_scores
import matplotlib.pyplot as plt
import os


def evaluate_LR(dtr, ltr, dte, lte, cfn, cfp):
    label = "$lambda$"
    p = [0.1, 0.5, 0.9]
    mins_train = [[], [], []]
    mins_test = [[], [], []]
    k = 0
    values = np.logspace(-5, 5, 31)
    for lam in values:
        k += 1
        print(f"iterazione {k}")
        model = LR(lam, 0.5)
        train_scores = np.load(f"score_models/LR/LR_l_{lam}_pt_0.5.npy")
        test_scores = get_evaluation_scores(dtr, ltr, dte, lte, model, model_desc=model.description())
        for j in range(len(p)):
            mins_train[j].append(compute_min_dcf(train_scores, ltr, p[j], cfn, cfp))
            mins_test[j].append(compute_min_dcf(test_scores, lte, p[j], cfn, cfp))

    for i in range(len(mins_train)):
        plt.plot(values, mins_train[i], label=f"eff_p={p[i]}_train", linestyle="dotted")
        plt.plot(values, mins_test[i], label=f"eff_p={p[i]}_test")
    plt.xscale("log")
    plt.xlabel(label)
    plt.ylabel("minDCF")
    plt.xlim([values[0], values[-1]])
    plt.legend()
    if not os.path.exists("figures/evaluation"):
        os.makedirs("figures/evaluation")
    plt.savefig("figures/evaluation/LR1")
