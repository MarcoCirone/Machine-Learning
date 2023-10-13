import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("./")
from general.utils import k_fold

def hist(d, l):
    for i in range(d.shape[0]):
        plt.figure()
        for c in range(l.max() + 1):
            x = d[i, l == c]
            plt.hist(x.reshape(x.size, ), alpha=0.4, label="c")
        plt.xlabel(f'Dimension {i + 1}')
        plt.legend()
        plt.show()


def scatter_2d(d, l):
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            if i != j:
                plt.figure()
                for c in range(l.max() + 1):
                    d_c = d[:, l == c]
                    x1 = d_c[i, :]
                    x2 = d_c[j, :]
                    plt.scatter(x1, x2, label=c)
                plt.xlabel(f'Dimension {i}')
                plt.xlabel(f'Dimension {j}')
                plt.legend()


def scatter_3d(d, l):
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            for k in range(d.shape[0]):
                if i != j and i != k and j != k:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for c in range(l.max() + 1):
                        d_c = d[:, l == c]
                        x = d_c[i, :]
                        y = d_c[j, :]
                        z = d_c[k, :]
                        ax.scatter(x, y, z, label=c)
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')
                    plt.legend()
def plot_min_dcfs(dtr, ltr, cfn, cfp, model, pt=None, svm_params=None, reg_term=None, pca_m=None, seed=0):
    values = np.logspace(-5, 5, num=31)
    label = "C"
    min_dcf_list = []
    e = [0.5, 0.1, 0.9]
    k = 0

    if svm_params is None:
        values = np.logspace(-5, 5, num=51)
        label = "$lambda$"

    # for eff_p in e:
    #     mins = []
    #     for v in values:
    #
    #         if svm_params is None:
    #             reg_term = v
    #         else:
    #             svm_params[1] = v
    #
    #         dcf_min = k_fold(dtr, ltr, 5, model, eff_p, cfn, cfp, seed=seed, pt=pt, reg_term=reg_term, svm_params=svm_params, pca_m=pca_m)
    #         mins.append(dcf_min)
    #         k += 1
    #         print(f"Iterazione {k}: prior= {eff_p} {label}= {v} => min_dcf= {dcf_min}")
    #     min_dcf_list.append(mins.copy())

    min_dcf_list = [
        [
    0.34503968253968254,
    0.19523809523809524,
    0.1523809523809524,
    0.135515873015873,
    0.12658730158730158,
    0.12083333333333333,
    0.11865079365079365,
    0.11805555555555555,
    0.12023809523809524,
    0.11825396825396825,
    0.11865079365079365,
    0.11845238095238095,
    0.11904761904761904,
    0.11865079365079365,
    0.11964285714285713,
    0.12123015873015873,
    0.12123015873015874,
    0.12222222222222222,
    0.13134920634920635,
    0.2642857142857143,
    0.16865079365079366,
    0.4876984126984127,
    0.43154761904761907,
    0.5928571428571429,
    0.5259920634920635,
    0.6672619047619047,
    0.6557539682539683,
    0.709920634920635,
    0.602579365079365,
    0.6136904761904762,
    0.6017857142857143
        ],
        [
    0.661904761904762,
    0.4172619047619048,
    0.3494047619047619,
    0.3482142857142857,
    0.3202380952380952,
    0.3047619047619048,
    0.29523809523809524,
    0.30357142857142855,
    0.2922619047619048,
    0.30952380952380953,
    0.305952380952381,
    0.29642857142857143,
    0.2988095238095238,
    0.2946428571428571,
    0.29166666666666663,
    0.3,
    0.2994047619047619,
    0.3071428571428571,
    0.31785714285714284,
    0.43988095238095243,
    0.36607142857142855,
    0.8863095238095239,
    0.9666666666666668,
    0.9988095238095238,
    0.9636904761904761,
    0.9583333333333334,
    0.9940476190476191,
    0.993452380952381,
    0.9952380952380953,
    0.8845238095238096,
    0.9976190476190477
        ],
        [
    0.6712301587301588,
    0.6005952380952381,
    0.49940476190476196,
    0.44305555555555554,
    0.40436507936507937,
    0.3567460317460318,
    0.33968253968253975,
    0.35039682539682543,
    0.34722222222222227,
    0.3380952380952381,
    0.34186507936507937,
    0.343452380952381,
    0.339484126984127,
    0.3400793650793651,
    0.34246031746031746,
    0.3492063492063493,
    0.341468253968254,
    0.3464285714285714,
    0.3890873015873016,
    0.9496031746031746,
    0.44325396825396834,
    0.9734126984126985,
    0.9763888888888888,
    0.9589285714285714,
    0.8890873015873016,
    0.9444444444444443,
    0.9952380952380953,
    0.9970238095238096,
    0.9720238095238096,
    0.9972222222222222,
    0.9958333333333333
        ]

    ]
    for i in range(len(min_dcf_list)):
        plt.plot(values, min_dcf_list[i], label=f"eff_p={e[i]}")
    plt.xscale("log")
    plt.xlabel(label)
    plt.ylabel("minDCF")
    plt.xlim([values[0], values[-1]])
    plt.legend()
    plt.show()
