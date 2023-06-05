import seaborn as sns

def set(context='paper', gridstyle='whitegrid', linecolorstyle='bright', font_scale=1.2):
    sns.set(context, gridstyle,  linecolorstyle, font_scale=font_scale,
            rc={"font.size": 9.0, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
                "lines.linewidth": 1.0, 'grid.linestyle': '--',
                'font.family': 'serif', 'font.serif': 'Times New Roman'})

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(1, 100, 100)
    y = x

    set(gridstyle='darkgrid', linecolorstyle='bright')
    plt.subplot(121)
    plt.plot(x, y)
    plt.gca().set_aspect('equal', adjustable='box')

    set(gridstyle='whitegrid', linecolorstyle='dark')
    plt.subplot(122)
    plt.plot(x, y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
