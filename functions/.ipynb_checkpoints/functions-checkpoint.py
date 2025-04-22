import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.integrate import quad
from scipy.interpolate import interp1d

#######################################
# List of color-blind-friendly colors #
#######################################

cols = ['k', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
# Also set the axis to be behind all line plots
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.size'] = 14

#################################
### GMM, potential, and drift ###
#################################

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

#################################
### Auto-correlation function ###
#################################

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

##################################
### Scatter-plot and histogram ###
##################################
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

###################################
### Local dimension computation ###
###################################

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
    weights = (1/K)
    return np.sum( increments.T * weights, axis=1 )**(-1)


########################################
    ################################
### Utilitaries for the illustration ###
    ################################
########################################

# Uniform distribution
def generate_uniform( N=10**3 , seed=None ):
    rng = np.random.default_rng(seed=seed)
    rv = rng.uniform()
    X = rng.uniform( -1 , 1 , N )
    Y = rng.uniform( -1 , 1 , N )
    return X, Y

# Utilitary to put back data into [-1,+1]
def back_in_square(x):
    x_shift = np.array(x) + 1
    x_shift_modulo = x_shift % 2
    x_back = x_shift_modulo - 1
    return x_back
    
# Random numbers close to x=y line
def generate_line( N=10**3 , eps=1e-1 , seed=None ):
    # Generate points on x=y line between -1 and +1
    rng = np.random.default_rng(seed=seed)
    X = rng.uniform( -1 , 1 , N )
    Y = X.copy()

    # Add gaussian perturbation
    X += eps * rng.normal(size=N)
    Y += eps * rng.normal(size=N)

    # Put back in square
    X = back_in_square(X)
    Y = back_in_square(Y)

    return X, Y

# Random Gaussian-like generator
def generate_gaussian_like(N=1000, sigma=0.5, sign=1,  seed=None):
    """
    Generate N points within [-1,1] x [-1,1] where:
    - R follows a probability density ∝ exp( +- R^2 / sigma^2) * R for 0 < R < sqrt(2)
    - Theta follows a uniform distribution U(0, 2π)
    
    Parameters:
    - N (int): Number of points to generate
    - sigma (float): Shape parameter controlling spread of points
    - sign (-1 or +1): To allow for increasing or decreasing function of R
    - seed (int, optional): Random seed for reproducibility
    
    Returns:
    - x_vals, y_vals: Arrays of sampled points
    """
    if sigma == 0:
        raise ValueError("Sigma cannot be zero.")
    
    if not sign in [-1,1]:
        raise ValueError("Sign must be +1 or -1.")

    rng = np.random.default_rng(seed=seed)

    # Step 1: Precompute the CDF of the distribution
    r_max = np.sqrt(2)
    r_vals = np.linspace(0, r_max, 1000)
    pdf_vals = np.exp( sign * r_vals**2 / sigma**2) * r_vals
    normalization, _ = quad(lambda r: np.exp(sign * r**2 / sigma**2) * r, 0, r_max)
    cdf_vals = np.cumsum(pdf_vals) / np.sum(pdf_vals)  # Approximate normalized CDF

    # Create an interpolation function for inverse transform sampling
    inverse_cdf = interp1d(cdf_vals, r_vals, bounds_error=False, fill_value=(0, r_max))

    x_vals, y_vals = [], []

    while len(x_vals) < N:
        # Step 2: Sample R using inverse transform sampling
        U = rng.uniform(0, 1, size=N)
        r = inverse_cdf(U)

        # Step 3: Sample Theta uniformly
        theta = rng.uniform(0, 2*np.pi, size=N)

        # Step 4: Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Step 5: Keep only points inside the square [-1,1] x [-1,1]
        mask = (np.abs(x) <= 1) & (np.abs(y) <= 1)
        x_vals.extend(x[mask])
        y_vals.extend(y[mask])

    # Ensure exactly `N` points are returned
    return np.array(x_vals[:N]), np.array(y_vals[:N])


### Generate plots for comparisons ###

def compare_plots(seed=1312, N=4*10**3, M=100, K = 600,
                  eps=.025, sigma_pos=.6, sigma_neg=.3,
                   alpha1=.3, alpha2=.2, orientation='horizontal', savefig=False, 
                  path_save='./', name_save='fig_compare', dpi=100):

    plt.figure(
        figsize = [ (7,14) , (14,6) ][int(orientation=='horizontal')] )

    for i_type in range(4):

        if i_type == 0:
            data = np.reshape( generate_uniform( seed=seed, N=N*M ) , (M,N,2) )
            distribution_type = 'uniform 2D'
        elif i_type == 1:
            data = np.reshape( np.array( generate_line( eps=eps , seed=seed, N=N*M ) ).T , (M,N,2) )
            distribution_type = 'uniform 1D'
        elif i_type == 2:
            data = np.reshape( generate_gaussian_like( sigma=sigma_pos , sign=+1 , seed=seed, N=N*M ) , (M,N,2) )
            distribution_type = 'density-minimum 2D'
        elif i_type == 3:
            data = np.reshape( generate_gaussian_like( sigma=sigma_neg , sign=-1 , seed=seed, N=N*M ) , (M,N,2) )
            distribution_type = 'density-maximum 2D'
        
        # Compute distances to zero based on input "data"
        dist = np.sum( np.array(data)**2 , axis=-1 ) ** .5
        ind_sorted = np.argsort(dist, axis=1)
        dist_sorted = np.take_along_axis(dist, ind_sorted, axis=1)
        dist_ana = dist[:,:K]
        dim = locdim(dist_ana)
        dim_all = np.full( (M,N) , np.nan )
        for m in range(M):
            dist_all = dist_sorted[m][dist_sorted[m]<1]
            for i in range(3,len(dist_all)):
                dim_all[m,i] = locdim(dist_all[:i].reshape(1,-1))[0]
        dim_low = np.quantile(dim_all, 0.025, axis=0)
        dim_med = np.median(dim_all, axis=0)
        dim_high = np.quantile(dim_all, 0.975, axis=0)

        if orientation == 'horizontal':
            plt.subplot(240+[1,2,5,6][i_type])
        elif orientation == 'vertical':
            plt.subplot(420+[1,3,5,7][i_type])
        plt.scatter([0,0], [0,0], marker='+', color='k', zorder=1)
        plt.scatter(data[m,:,0], data[m,:,1], s=.2, zorder=10, color='k', alpha=alpha1)
        plt.scatter(data[m,:,0][ind_sorted[m,:K]], data[m,:,1][ind_sorted[m,:K]], s=.2, zorder=10, color='r')
        # add a text with the estimated dimension
        plt.text( -.3 , .7 , r'$\hat{d}=$' + str(dim_med[K])[:4], zorder=20,
                    bbox=dict(facecolor='1', alpha=.8, linewidth=.1, edgecolor='k'))
        # add a text with distribution type
        plt.text( -.93 , -.9 , distribution_type, zorder=20,
                 bbox=dict(facecolor='1', alpha=.8, linewidth=.1, edgecolor='k'))
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        if orientation == 'horizontal':
            plt.title('(' + ['a','b','e','f'][i_type] + ')')
        elif orientation == 'vertical':
            plt.title('(' + ['a','c','e','g'][i_type] + ')')
        

        
        if orientation == 'horizontal':
            plt.subplot(240+[3,4,7,8][i_type])
        elif orientation == 'vertical':
            plt.subplot(420+[2,4,6,8][i_type])
        plt.grid()
        plt.plot( np.arange(1,N+1) , [int(distribution_type[-2])]*N , '--', color='.5', zorder=0 )
        plt.scatter( np.arange(1,K+1) , dim_med[:K] , s=.2, color='r')
        plt.scatter( np.arange(K+2,N+1) , dim_med[K+1:N] , s=.2, color='k')
        plt.fill_between( np.arange(1,K+1) , dim_low[:K] , dim_high[:K] , color='r', alpha=alpha2)
        plt.fill_between( np.arange(K+2,N+1) , dim_low[K+1:N] , dim_high[K+1:N] , color='k', alpha=alpha2)
        # add a text with distribution type
        plt.text( 5 , .25 , distribution_type, zorder=20,
                 bbox=dict(facecolor='1', alpha=1, linewidth=.1, edgecolor='k'))
        plt.xscale('log')
        plt.xlim([4, 3e3]); plt.ylim([0,5]); plt.yticks(np.arange(6))
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()
        plt.xlabel('number of analogues')
        plt.ylabel('estimated dimension')
        if orientation == 'horizontal':
            if i_type==0 or i_type==2:
                plt.gca().axes.yaxis.set_ticklabels([])
                plt.ylabel('')
            if i_type==0 or i_type==1:
                plt.gca().axes.xaxis.set_ticklabels([])
                plt.xlabel('')
            plt.title('(' + ['c','d','g','h'][i_type] + ')')
        elif orientation == 'vertical':
            plt.title('(' + ['b','d','f','h'][i_type] + ')')
            if not i_type == 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                plt.xlabel('')
        

    
    if savefig:
        plt.savefig(path_save + name_save + '.png', bbox_inches='tight', dpi=dpi )
    
    plt.show()
