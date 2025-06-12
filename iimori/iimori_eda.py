# -*- coding: utf-8 -*-
"""
Iimori Data Exploration, 6/11/25
"""

#libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#loading data
df = pd.read_excel('ROUTE_proteinuria_dataset.xlsx')
df['delta_eGFR'] = df['eGFR']-df['eGFR(last visit)'] #creating column with overall change in eGFR between first and last measurement (negative values incidicate an increase in eGFR)
df['slope'] = df['delta_eGFR']/df['observational duration'] #column with slope of change in eGFR over change in time


# the df_for_pca vaiable is a submatrix of the original data matrix (the second one only contains quantitative variables)
#df_for_pca = df[['gender', 'age', 'BMI', 'etiology of CKD','Cr','eGFR','UPCR','eGFR(last visit)','delta_eGFR','observational duration']].copy()
df_for_pca = df[['age', 'eGFR','eGFR(last visit)','delta_eGFR','observational duration']].copy()

if __name__=="__main__":

    df_for_plot = PCA(n_components=3).fit_transform(df_for_pca)

    #plotting results of PCA
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=0, azim=180)

    scatter = ax.scatter(
        df_for_plot[:, 0],
        df_for_plot[:, 1],
        df_for_plot[:,2],
        s=40,
        c=df['delta_eGFR']
    )


    ax.set(
        title="First three PCA dimensions",
        xlabel="Component 1",
        ylabel="Component 2",
        zlabel="Component 3",
    )

    #fig.savefig()

    '''
    plt.title("PCA of Iimori CKD Data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
    '''    








