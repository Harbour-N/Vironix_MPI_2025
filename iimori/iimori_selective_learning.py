import iimori_timeseries as it
from matplotlib import pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import cluster
from sklearn import mixture

plt.rcParams.update({'font.size': 14})

#
import iimori_eda as ie


## Conceptually:
# 1. collect a set of features physicians could look at
# 2. select a handful of landmarks/positions in feature space 
#    and weight selection of data into two (or more) pieces; so that 
#    two discriminators (physicians) get trained.preferentially on their own thing 
#    (e.g. mixture model with 1 component; fancier tools can apply for this selected training data)
# 3. From witheld data, they tell us what is real and what is not.
#

features = ['age', 'eGFR','eGFR(last visit)','slope','observational duration', 'Cr']


nphys = 3


df = ie.df[features]
mask = df.notna().all(axis=1)
df = df[mask] # keep only full records (no nan allowed)

X = df.values

km = cluster.KMeans(n_clusters=nphys) # 2 physicians
km.fit(X)
centers=km.cluster_centers_
radius = 30 # ad-hoc; work out a nice number later
phys_threshold = -15 # if log-likelihood below this; we call it synthetic (out of distribution)


# training purposes only: physicians deciding what is real.
D = metrics.pairwise_distances(X, centers)
#keeps = D

train_mask = D < radius # points only within Euclidean distance "radius" get fed to physician j

# 

# Train physicians to expect certain groups of patients.
physicians = [mixture.GaussianMixture(n_components=1) for _ in range(nphys)]
for i in range(nphys):
    phys = physicians[i]
    phys.fit(X[train_mask[:,i]])
    physicians[i] = phys
    for j in range(nphys):
        mine = phys.score_samples(X[train_mask[:,j]]) > phys_threshold
        print(f'{i} given {j} patients...', np.where(mine)[0])
        #print(f'phys. {i} log-likelihoods for phys {j} patients: ',phys.score_samples(X[train_mask[:,j]]))
    #print(X[train_mask[:,rest]] < phys_threshold) #TODO: is "1" to mean "synthetic"?

# Illustrating biased opinions
fig,ax = plt.subplots(1, nphys, figsize=(nphys*4+1,4), sharex=True, sharey=True, constrained_layout=True)

pca = decomposition.PCA(n_components=2)
X_proj = pca.fit_transform(X)

phys_pca = pca.transform([physicians[i].means_[0] for i in range(nphys)])
for i in range(nphys):
    # patients categorized of realism by log-likelihood based on physician's training
    cax=ax[i].scatter(X_proj[:,0], X_proj[:,1], c=physicians[i].score_samples(X), cmap=plt.cm.cividis, vmin=phys_threshold-1, vmax=phys_threshold)
    
    # Physician's center of focus
    ax[i].scatter(phys_pca[i,0], phys_pca[i,1], marker='o', fc='#f00', s=200, alpha=0.5)
    
    ax[i].set(xlabel='PC1', ylabel='PC2')
    ax[i].text(0,1,f'Physician {i}', transform=ax[i].transAxes, ha='left', va='top', bbox={'fc':'w', 'ec':'k'})
    if i==nphys-1:
        fig.colorbar(cax)
    ax[i].grid(True)

fig.suptitle('Iimori: log-likelihoods for physician classification of "synthetic" data')

fig.savefig('../output/iimori_elephant_gaussians.png')
fig.savefig('../output/iimori_elephant_gaussians.pdf')
fig.show()
