
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import digamma, gamma
from numpy import linalg as la
from matplotlib.patches import Ellipse

#import gaussian_mixtures
from utils.Gaussian_Mixtures import GaussianMixture
from utils.Confidence_Ellipse import plot_confidence_ellipse




# Bivariate example
dim = 2

# Settings
n =600
NumberOfMixtures = 5

# Mixture weights (non-negative, sum to 1)
# here we use the same weights as the quoted publication
w = [0.156572, 0.18465, 0.163462, 0.315405, 0.179911]

# Mean vectors and covariance matrices
# here we use the same means and covariance matrices as the quoted publication
MeanVectors = [ [0,0], [3,-3], [3,3], [-3,3] ,[-3,-3] ]
CovarianceMatrices = [ [[1, 0], [0, 1]], [[1, 0.5], [0.5, 1]], [[1, -.5], [-.5, 1]]
    ,[[1,0.5],[0.5,1]] ,[[1,-0.5],[-0.5,1]] ]


################################################### Draw samples
# Initialize arrays
samples = np.empty( (n,dim) )
samples[:] = np.NaN
componentlist = np.empty( (n,1) )
componentlist[:] = np.NaN

# Generate samples
for iter in range(n):
    # Get random number to select the mixture component with probability according to mixture weights
    DrawComponent = random.choices(range(NumberOfMixtures), weights=w, cum_weights=None, k=1)[0]
    # Draw sample from selected mixture component
    DrawSample = np.random.multivariate_normal(MeanVectors[DrawComponent], CovarianceMatrices[DrawComponent], 1)
    # Store results
    componentlist[iter] = DrawComponent
    samples[iter, :] = DrawSample

# Report fractions
print('Fraction of mixture component 0:', np.sum(componentlist==0)/n)
print('Fraction of mixture component 1:',np.sum(componentlist==1)/n)
print('Fraction of mixture component 2:',np.sum(componentlist==2)/n)
print('Fraction of mixture component 3:',np.sum(componentlist==3)/n)
print('Fraction of mixture component 4:',np.sum(componentlist==4)/n)




########################################################
# Visualize result
plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
plt.grid()
plt.xlabel('x-axis')
plt.xlabel('y-axis')
plt.show()

###########################################################
#### initialize the model
mixture_model=GaussianMixture(data=samples, number_gaussian=20, max_iter=100,display=False,save=True,name='mix_of_five',dir='mix_of_five_img',save_freq=100)
# fit the model
mixture_model.fit_predict()

# Display the ELBO
mixture_model.display_elbo()

"""
# print the final results
print(f"Estimation of the Weights {mixture_model.weight_estimation:.5f}")

print(f"Estimation of the Covariance Matrix {mixture_model.covariance_matrix:.5f}")

print(f"Estimation of the Mean {mixture_model.means:.5f}")
"""

