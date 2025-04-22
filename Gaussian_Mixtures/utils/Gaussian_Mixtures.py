import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import digamma, logsumexp
from scipy.special import gamma
from numpy import linalg as la
from matplotlib.patches import Ellipse
from .Confidence_Ellipse import plot_confidence_ellipse
import os

class GaussianMixture():
    """
    Gaussian Mixture model.
    Python implementation of the paper
    Corduneanu, Bishop, Variational Bayesian Model Selection for Mixture Distributions,
    Artificial Intelligence and Statistics, 2001 T. Jaakkola and T. Richardson (Eds), 27-34,
    Morgan Kaufmann

    Thanks to Julien Nonin (https://github.com/juliennonin). The confidence ellipse
    display plots are primarily based on his code, with small modifications.
    This appears in the _display_intermidiary_results() method and
    the confidence_Ellipses
    """


    def __init__(self, data, number_gaussian, max_iter=400, beta=1,display=False,save=False,name=None,dir=None,save_freq=10):
        self.data = data
        self.M = number_gaussian
        self.max_iter = max_iter
        self.beta = beta
        self.display=display
        self.save=save
        self.name=name
        self.dir=dir
        self.save_freq=save_freq

    def _save_location(self,saveit,dirit):
        if saveit ==True:
            if not os.path.exists(dirit):
                os.makedirs(dirit)

    def _initialize(self):
        # Normalize the data
        self.X = self.data - [[np.mean(self.data[:, 0]), np.mean(self.data[:, 1])]]
        self.X = self.X / [[np.std(self.X[:, 0]), np.std(self.X[:, 1])]]

        self.N, self.d = self.X.shape
        self.nu = self.d
        self.pi = np.ones((self.M)) / self.M
        self.pin = 0.5 * np.random.rand(self.M, self.N)
        for i in range(self.M):
            for n in range(self.N):
                self.pin[i, n] /= ((np.sum(self.pin, axis=0))[n])

        self.means = (-2 + 4 * np.random.rand(self.M, 2))  # np.zeros((M,2))
        self.Cov = np.cov(self.X.T)
        self.V = self.Cov  # np.linalg.inv(Cov)#np.array([[1.05438334, -0.17315909],[-0.17315909, 1.07220622
        self.x_T = []

        for i in range(self.M):
            self.rand = np.random.rand()
            self.x_T.append(np.eye(self.d) + [[0, 0.9 * (-1 + 2 * self.rand)], [0.9 * (-1 + 2 * self.rand), 0]])

        self.nu_T = self.nu * np.ones(self.M)
        self.mmu = self.means
        self.V_T = []
        self.T_mu = []
        for i in range(self.M):
            self.V_T.append(self.V)
            self.T_mu.append(self.beta * np.eye(self.d) + self.x_T[i] * (self.pin.sum(axis=1)[i]))

    def _compute_expectations(self):
        # Compute the expectation values of the parameters of the model
        self.x_mu_mu_T = []  # Expectation value of mu mu^T, <\mu \mu^T>
        self.x_mu = self.mmu  # Expectation value of mu , <\mu>
        self.x_ln_T = []  # Expectation value: <\ln |T| >
        self.x_muT_mu = []  # Expectation value <\mu^T \mu>
        self.x_T = []
        for i in range(self.M):
            self.x_T.append(self.nu_T[i] * np.linalg.inv(self.V_T[i]))
            self.x_mu_mu_T.append(np.linalg.inv(self.T_mu[i]) + np.outer(self.mmu[i], self.mmu[i]))
            self.x_ln_T.append(self.d * np.log(2) - np.log(np.linalg.det(self.V_T[i])) + digamma(
                0.5 * (self.nu_T[i] - np.arange(self.d))).sum())
            self.x_muT_mu.append(self.mmu[i] @ self.mmu[i] + np.trace(np.linalg.inv(self.T_mu[i])))

    def _compute_p_tilde(self):
        # sub-routine to compute p_tilde

        self.p_tilde = np.zeros((self.M, self.N))
        for i in range(self.M):
            for n in range(self.N):
                self.p_tilde[i, n] = np.exp(0.5 * self.x_ln_T[i] + np.log(self.pi[i]) - 0.5 * np.trace(
                    np.matmul(self.x_T[i],
                              np.outer(self.X[n], self.X[n]) - np.outer(self.x_mu[i], self.X[n]) - np.outer(self.X[n],
                                                                                                            self.x_mu[
                                                                                                                i]) +
                              self.x_mu_mu_T[i])))

    def _compute_conditional_probabilities(self):

        self.x_ln_P_mu_T_s = 0
        for i in range(self.M):
            for n in range(self.N):
                self.x_ln_P_mu_T_s += self.pin[i, n] * (
                            0.5 * self.x_ln_T[i] - 0.5 * self.d * np.log(2 * np.pi) - 0.5 * np.trace(self.x_T[i] @ (
                                np.outer(self.X[n], self.X[n]) - np.outer(self.x_mu[i], self.X[n]) - np.outer(self.X[n],
                                                                                                              self.x_mu[
                                                                                                                  i]) +
                                self.x_mu_mu_T[i])))

        # for i in range(self.M):

        self.x_ln_P_s = np.sum(np.log(self.pi) @ self.pin)
        self.x_ln_P_mu = self.M * self.d * np.log(self.beta / (2 * np.pi)) - 0.5 * self.beta * np.sum(self.x_muT_mu)
        self.x_ln_P_T = self.M * (
                    -0.5 * self.nu * self.d * np.log(2) - 0.25 * self.d * (self.d - 1) * np.log(np.pi) - np.sum(
                np.log(gamma(0.5 * (self.nu + np.arange(self.d))))) + 0.5 * self.nu * np.log(np.linalg.det(self.V))) + (
                                    0.5 * (self.nu - self.d - 1) * np.sum(self.x_ln_T) - 0.5 * np.sum(
                                np.trace(self.V @ self.x_T)))
        self.x_ln_Q_s = np.sum((self.pin * np.log(self.pin)))
        self.x_ln_Q_mu = np.sum(-0.5 * self.d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(self.T_mu)))
        self.x_ln_Q_T = 0
        for i in range(self.M):
            self.x_ln_Q_T += (-0.5 * self.nu_T[i] * self.d * np.log(2) - 0.25 * self.d * (self.d - 1) * np.log(
                np.pi) - np.sum(np.log(gamma(0.5 * (self.nu_T[i] + np.arange(self.d))))) + 0.5 * self.nu_T[i] * np.log(
                np.linalg.det(self.V_T[i]))) + (0.5 * (self.nu_T[i] - self.d - 1) * self.x_ln_T[i] - 0.5 * np.trace(
                self.V_T[i] @ self.x_T[i]))
        return self.x_ln_P_mu_T_s + self.x_ln_P_s + self.x_ln_P_mu + self.x_ln_P_T - self.x_ln_Q_s - self.x_ln_Q_mu - self.x_ln_Q_T  # return the elbo

    def _display_intermidiary_results(self,v,s,trial_index):
        plt.figure(figsize=(6, 6))
        plt.plot(*self.X.T, 'o', c='b', alpha=0.5)
        ax = plt.gca()
        for k in range(self.M):
            if self.pi[k] >= 1 / (2 * self.M):
                plot_confidence_ellipse(self.mmu[k], np.linalg.inv(self.x_T[k]), 0.95, ax=ax, ec='teal')
        ax.set_aspect('equal')
        if v == True:
            plt.show()
        if v == False and s == True:
            plt.savefig(self.dir+'/'+ self.name+ str(trial_index) + '.jpg')
            plt.close()


    def _update_parameters(self):
        self._compute_p_tilde()
        for i in range(self.M):
            for n in range(self.N):
                self.pin[i, n] = self.p_tilde[i, n] / ((np.sum(self.p_tilde, axis=0)[n]))

        self.T_mu = []
        self.mmu = []
        self.nu_T = []
        self.V_T = []
        for i in range(self.M):
            self.T_mu.append(self.beta * np.eye(self.d) + self.x_T[i] * (self.pin.sum(axis=1)[i]))
            self.mmu.append(np.linalg.inv(self.T_mu[i]) @ self.x_T[i] @ (self.pin[i] @ self.X))
            self.nu_T.append(self.nu + np.sum(self.pin, axis=1)[i])
            self.v_temp = np.zeros((self.d, self.d))
            for n in range(self.N):
                self.v_temp += self.pin[i, n] * (
                            np.outer(self.X[n], self.X[n]) - np.outer(self.x_mu[i], self.X[n]) - np.outer(self.X[n],
                                                                                                          self.x_mu[
                                                                                                              i]) +
                            self.x_mu_mu_T[i])
            self.V_T.append(self.V + self.v_temp)

    def fit_predict(self):

        self._initialize()
        self._save_location(self.save, self.dir)

        self.elbo = [1]
        for trials in range(self.max_iter):
            self._compute_expectations()
            self.elbo.append(self._compute_conditional_probabilities())
            self._update_parameters()
            self.pi = np.sum(self.pin, axis=1) / self.N
            # for i in range(self.M):
            #   if np.abs(self.pi[i])<10**-6:
            #      self.pi[i]=0

            if trials % (self.max_iter / self.save_freq) == 0:
                self._display_intermidiary_results(self.display,self.save,trials)
                print(f'Evidence lower bound: {self.elbo[-1]:.4f}, step {trials}')
            if np.abs(self.elbo[trials + 1] - self.elbo[trials]) < 10 ** (-15):
                break
        self._get_final_parameters()
        self._display_final_results()

    def _get_final_parameters(self):
        self.final_pi=[]
        self.final_x_T=[]
        self.final_mmu=[]
        for i in range(self.M):
            if self.pi[i]>10**-12:
                self.final_pi.append(self.pi[i])
                self.final_x_T.append(self.x_T[i])
                self.final_mmu.append(self.mmu[i])

        self.weight_estimation = (self.final_pi)
        self.covariance_matrix=np.linalg.inv(self.final_x_T)
        self.means=self.final_mmu

    def display_elbo(self):
        plt.figure(figsize=(6, 6))
        plt.plot(np.arange(1, len(self.elbo)), self.elbo[1:])
        plt.ylabel('ELBO')
        plt.show()


    def _display_final_results(self):
        plt.figure(figsize=(6, 6))
        plt.plot(*self.X.T, 'o', c='b', alpha=0.5)
        ax = plt.gca()
        for k in range(self.M):
            if self.pi[k] >= 1 / (2 * self.M):
                plot_confidence_ellipse(self.mmu[k], np.linalg.inv(self.x_T[k]), 0.95, ax=ax, ec='teal')
        ax.set_aspect('equal')
        plt.xlabel('x-axis')
        plt.xlabel('y-axis')
        plt.title('Confidence Ellipse of the Gaussian Mixture Model')
        plt.show()

