import numpy as np
from tqdm.notebook import tqdm
from scipy.stats import multivariate_normal as mvn

# list of color-blind-friendly colors
cols = ['k', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

### GMM, potential, and drift ###

def density_gmm(x, means, covs, weights):
    gaussians = []
    for i in range(len(weights)):
        rv = mvn(means[i], covs[i])
        gaussians.append( rv.pdf(x) )
    gaussians = np.array( gaussians )
    return np.sum(gaussians.T*weights, axis=-1)

def potential_gmm(x, means, covs, weights):
    return - .5 * np.log( density_gmm(x , means , covs , weights ) )

def drift_gmm(x, means, covs, weights):
    dx = 1e-4
    V = potential_gmm(x, means , covs , weights )
    drift = np.zeros( x.T.shape )
    for i in range(x.shape[-1]):
        dX = np.zeros( x.shape[-1] , dtype='float64' )
        dX[i] = dx
        drift[i] = - ( potential_gmm(x + dX , means , covs , weights ) - V ) / dx
    return drift.T

### Auto-correlation function ###

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

### Scatter-plot and histogram ###
# note: this code is a modification of https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html, last accessed 05 Déc. 2024

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s = 2, alpha=.3, color='k' )

    # the density plot
    bins = 50
    ax_histx.hist(x, bins=bins, color=cols[1])
    ax_histx.hist(y, bins=bins, histtype='step', color=cols[2])
    ax_histy.hist(y, bins=bins, orientation='horizontal', color=cols[2])
    ax_histy.hist(x, bins=bins, histtype='step', orientation='horizontal', color=cols[1])    

### Local dimension computation ###

def locdim(dist):
    # Simple function to estimate the local dimension from the data surrounding a given 'query' or 'target' 
    # The function takes as argument:
    # 'dist' : analog-to-target distances, sorted, shape = (number of querys, number of analogs per query)
    # The function returns:
    # the estimate of local dimension, shape = (number of querys)
    if len(dist.shape)!=2:
        print('Error: reshape your distances : (number of queries, number of analogs per query)')
    
    K = dist.shape[1]
    logdist = np.log(dist)
        
    increments = - logdist[:,:-1].T + logdist[:,-1]
    weights = (1/K) * (np.arange(2,K+1)/K)
    return np.sum( increments.T * weights, axis=1 )**(-1)

