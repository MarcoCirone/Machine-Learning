import matplotlib.pyplot as plt


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
