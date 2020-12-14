import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans



def boston():
    boston=load_boston()
    bos=pd.DataFrame(boston.data)
    bos.col=boston.feature_names
    bos['PRICE'] = boston.target
    X=bos.drop('PRICE',axis=1)
    linereg=LinearRegression()
    linereg.fit(X,bos['PRICE'])
    #calculate coefficient of each factor
    coef=(np.fabs(linereg.coef_)).tolist()

    max_index=coef.index(max(coef))
    return boston.feature_names[max_index]


def elbow():
    winex, _ = load_wine(return_X_y=True)
    irisx, _ = load_iris(return_X_y=True)

    #sum of squared distance
    iris_dis = []
    wine_dis = []
    #loop
    for i in range(1, 11):
        #fit models
        iris = KMeans(n_clusters=i).fit(irisx)
        wine = KMeans(n_clusters=i).fit(winex)

        iris_dis.append(iris.inertia_)
        wine_dis.append(wine.inertia_)

    #plot results
    #iris data
    plt.plot(range(1, 11), iris_dis)
    plt.title("SSD for Iris dataset")
    plt.xlabel("Num of clusters")
    plt.ylabel("SSD")
    plt.show()

    #wine data
    plt.plot(range(1, 11), wine_dis)
    plt.title("SSD for Wine dataset")
    plt.xlabel("Num of clusters")
    plt.ylabel("SSD")
    plt.show()

    #From these graphes,we can see the decline rate of ssd decreases at cluster three.



if __name__ == '__main__':
    print(boston())
    elbow()