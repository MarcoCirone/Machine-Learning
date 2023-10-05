import matplotlib as plt
def hist(D, L):
    for i in range(D.shape[0]):
        plt.figure()
        for c in range(L.max( ) +1):
            x = D[i, L== c]
            plt.hist(x.reshape(x.size, ), alpha=0.4, label=c)
        plt.xlabel(f'Dimension {i + 1}')
        plt.legend()
        plt.show()

def scatter2D(D, L):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i != j:
                plt.figure()
                for c in range(L.max() + 1):
                    D_c = D[:, L == c]
                    x1 = D_c[i, :]
                    x2 = D_c[j, :]
                    plt.scatter(x1, x2, label=c)
                plt.xlabel(f'Dimension {i}')
                plt.xlabel(f'Dimension {j}')
                plt.legend()


def scatter3D(D, L):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            for k in range(D.shape[0]):
                if i != j and i != k and j != k:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for c in range(L.max() + 1):
                        D_c = D[:, L == c]
                        x = D_c[i, :]
                        y = D_c[j, :]
                        z = D_c[k, :]
                        ax.scatter(x, y, z, label=c)
                        ax.set_xlabel('X Label')
                        ax.set_ylabel('Y Label')
                        ax.set_zlabel('Z Label')
                    plt.legend()

