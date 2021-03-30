from math import log, pow, exp
import pandas as pd


global pas
pas = 0.01

global x, y
x = pd.read_csv("p.csv", header=None)
y = pd.read_csv("d.csv", header=None)




def f_x_y(alpha, beta, gamma, i):
    f = alpha*log(x[0][i]) + beta*log(x[1][i]) + gamma - log(y[i][0])
    return pow(f, 2)

def f_x_y_derivative_alpha(alpha, beta, gamma, i):
    f = alpha*log(x[0][i]) + beta*log(x[1][i]) + gamma - log(y[i][0])
    return 2 * f * log(x[0][i])

def f_x_y_derivative_beta(alpha, beta, gamma, i):
    f = alpha*log(x[0][i]) + beta*log(x[1][i]) + gamma - log(y[i][0])
    return 2 * f * log(x[1][i])

def f_x_y_derivative_gamma(alpha, beta, gamma, i):
    f = alpha*log(x[0][i]) + beta*log(x[1][i]) + gamma - log(y[i][0])
    return 2 * f

def prediction(alpha, beta, gamma, X):
    return alpha*log(X[0]) + beta*log(X[1]) + gamma

def Phi(alpha, beta, gamma):
    somme = 0
    for i in range(x.shape[0]):
        somme += f_x_y(alpha, beta, gamma, i)
    return somme


def gradient_Phi(alpha, beta, gamma):
    # Alpha
    somme_alpha1 = 0
    for i in range(x.shape[0]):
        somme_alpha1 += f_x_y_derivative_alpha(alpha, beta, gamma, i)
    alpha_1 = alpha - pas*somme_alpha1
    # Beta
    somme_beta1 = 0
    for i in range(x.shape[0]):
        somme_beta1 += f_x_y_derivative_beta(alpha, beta, gamma, i)
    beta_1 = beta - pas*somme_beta1
    # Gamma
    somme_gamma1 = 0
    for i in range(x.shape[0]):
        somme_gamma1 += f_x_y_derivative_gamma(alpha, beta, gamma, i)
    gamma_1 = gamma - pas*somme_gamma1
    return alpha_1, beta_1, gamma_1


def main(alpha, beta, gamma):
    Phi0 = Phi(alpha, beta, gamma)
    alpha_1, beta_1, gamma_1 = gradient_Phi(alpha, beta, gamma)
    Phi1 = Phi(alpha_1, beta_1, gamma_1)
    n_iteration = 1
    
    while Phi1 < Phi0 and n_iteration < 10:
        n_iteration += 1
        Phi0 = Phi1
        alpha, beta, gamma = alpha_1, beta_1, gamma_1
        alpha_1, beta_1, gamma_1 = gradient_Phi(alpha, beta, gamma)
        Phi1 = Phi(alpha_1, beta_1, gamma_1)
    print(Phi1)
    return alpha, beta, gamma, n_iteration

alpha, beta, gamma, n_iteration = main(1, 1, 0)
print(exp(prediction(alpha, beta, gamma, [4, 5])))