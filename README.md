# Gaussian_Mixtures
A python implementation of Variational Bayesian Gaussian Mixture Models

Implements a python code from the algorithm described in:
Corduneanu, Bishop, Variational Bayesian Model Selection for Mixture Distributions,
Artificial Intelligence and Statistics, 2001 T. Jaakkola and T. Richardson (Eds), 27-34, Morgan Kaufmann

An example is displayed that is discussed in the paper of a mixture of 5 gaussians. 

<img_src="https://giphy.com/gifs/XLAN4qmyZDS6nZll6E" />


The code makes use of a confidence ellipse plot method that was created by 
Julien Nonin (https://github.com/juliennonin). The confidence ellipse
display plots are primarily based on his code, with small modifications.
This appears in the _display_intermidiary_results() method and the confidence_Ellipses
