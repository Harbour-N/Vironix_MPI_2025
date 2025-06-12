import iimori_timeseries as it
from matplotlib import pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition

plt.rcParams.update({'font.size': 14})

df = it.df_orig # original, before trimming based on eGFR missingness.

##
# Question 1: what linear model is a good characterization of people's CKD stage?
# When does prediction of CKD stage fail based on a small suite of biomarkers?

# Leaving out later timepoints to mitigate excessive removal of data due 
# to missingness. TODO: incorporate death and/or built-in features related 
# to length of stay into model.
predictors=[
    'eGFR(0M)',
    'eGFR(6M)',
    'eGFR(12M)',
#    'eGFR(18M)',
#    'eGFR(24M)',
#    'eGFR(30M)',
#    'eGFR(36M)',
    'Hb',
    'Alb',
    'Cr'
]

mask = df[predictors].notna().all(axis=1) # keep only data with no missingness.
X1 = df[predictors][mask]
y1 = df['CKD_stage'][mask]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X1, y1, test_size=0.5)

print(f'percentage data kept: {sum(mask)/len(mask)*100:.1f}')

lasso = linear_model.Lasso(alpha=0.05) # TODO: play with lasso params
lasso.fit(X_train, y_train)
pred_test = lasso.predict(X_test)
print('Lasso feature weights:', np.round(lasso.coef_, 2))
print('relevant features: ')
relevant_features = abs(lasso.coef_)>5e-3
print(np.array(predictors)[relevant_features])

mae_train = metrics.mean_absolute_error(y_train, lasso.predict(X_train))
mae_test = metrics.mean_absolute_error(y_test, pred_test)
print(f'mean abs err (train): {mae_train:.3f}')
print(f'mean abs err (test): {mae_test:.3f}')

# analyze most extreme errors in CKD stage prediction
test_errs = pred_test - y_test
tail_errs = abs(test_errs) > np.quantile(abs(test_errs),0.9)

#for i,b in enumerate(tail_errs):
#    if b:
#        print(X_test.iloc[i].T, y_test.iloc[i], pred_test[i])

pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
fig,ax = plt.subplots(1,2, figsize=(11,5), sharex=True, sharey=True, constrained_layout=True)
ax[0].scatter(X_pca[:,0], X_pca[:,1], c=y_test, cmap=plt.cm.viridis)
# put in a cute legend... very hacky
for i in range(5):
    ax[0].scatter([],[], color=plt.cm.viridis(i/(5-1)), label=f'Stage {i+1}')
cax = ax[1].scatter(X_pca[:,0], X_pca[:,1], c=test_errs, cmap=plt.cm.coolwarm, vmin=-1, vmax=1) # mark locations of worst predictions
fig.colorbar(cax)

for i in range(2):
    ax[i].set(xlabel='PC1', ylabel='PC2')
    ax[i].grid(True, zorder=-100)
ax[0].set_title('PCA; colored by CKD stage')
ax[1].set_title(f'PCA; prediction error (max: {max(abs(test_errs)):.2f})')

ax[0].legend(loc='upper right')

fig.savefig('../output/iimori_ckd_prediction_vis.png', bbox_inches='tight')
fig.savefig('../output/iimori_ckd_prediction_vis.pdf', bbox_inches='tight')
fig.show()

