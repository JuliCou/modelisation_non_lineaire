from math import pow, exp
import random
import pandas as pd
import numpy as np


global pas
pas = 0.1

global X, Y
X = pd.read_csv("p.csv", header=None)
Y = pd.read_csv("d.csv", header=None)



# j : num du neurones
# j=0 : P11
# j=1 : P12
def P_1j(X, coefficients, j, individu):
    x = coefficients[3*j] * X[0][individu]
    x += coefficients[3*j+1] * X[1][individu]
    x -= coefficients[3*j+2]
    return activation(x)


def P_2(x1, x2, coefficients):
    x = coefficients[3*2] * x1
    x += coefficients[3*2+1] * x2
    x -= coefficients[3*2+2]
    return activation(x)


def Neural_prediction(X, coefficients, individu):
    x1 = P_1j(X, coefficients, 0, individu)
    x2 = P_1j(X, coefficients, 1, individu)
    return P_2(x1, x2, coefficients)


def prediction(x1, x2, coefficients):
    x_11 = coefficients[0] * x1
    x_11 += coefficients[1] * x2
    x_11 -= coefficients[2]
    x_11 = activation(x_11)
    x_12 = coefficients[3*1] * x1
    x_12 += coefficients[3*1+1] * x2
    x_12 -= coefficients[3*1+2]
    x_12 = activation(x_12)
    return P_2(x_11, x_12, coefficients)


def activation(x):
    return exp(2*x) / (1 + exp(2*x))


def dev_activation(x):
    numerateur = 2*exp(2*x)*(1+exp(2*x)) - 2*exp(4*x)
    denominateur = pow(1+exp(2*x), 2)
    return numerateur / denominateur


def der_S_a111(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        x_der = coefficients[3*0] * X[0][individu]
        x_der += coefficients[3*0+1] * X[1][individu]
        x_der -= coefficients[3*0+2]
        f_der = dev_activation(x_der)
        t_terme = coefficients[3*2] * f_der * X[0][individu]
        somme += delta_prediction * f_der_x * t_terme
    return somme


def der_S_a112(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        x_der = coefficients[3*0] * X[0][individu]
        x_der += coefficients[3*0+1] * X[1][individu]
        x_der -= coefficients[3*0+2]
        f_der = dev_activation(x_der)
        t_terme = coefficients[3*2] * f_der * X[1][individu]
        somme += delta_prediction * f_der_x * t_terme
    return somme


def der_S_a121(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        x_der = coefficients[3*1] * X[0][individu]
        x_der += coefficients[3*1+1] * X[1][individu]
        x_der -= coefficients[3*1+2]
        f_der = dev_activation(x_der)
        t_terme = coefficients[3*2+1] * f_der * X[0][individu]
        somme += delta_prediction * f_der_x * t_terme
    return somme


def der_S_b11(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        x_der = coefficients[3*0] * X[0][individu]
        x_der += coefficients[3*0+1] * X[1][individu]
        x_der -= coefficients[3*0+2]
        f_der = dev_activation(x_der)
        t_terme = coefficients[3*2] * f_der
        somme += delta_prediction * f_der_x * t_terme
    return -1 * somme


def der_S_b12(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        x_der = coefficients[3*1] * X[0][individu]
        x_der += coefficients[3*1+1] * X[1][individu]
        x_der -= coefficients[3*1+2]
        f_der = dev_activation(x_der)
        t_terme = coefficients[3*2+1] * f_der
        somme += delta_prediction * f_der_x * t_terme
    return -1 * somme


def der_S_a122(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        x_der = coefficients[3*1] * X[0][individu]
        x_der += coefficients[3*1+1] * X[1][individu]
        x_der -= coefficients[3*1+2]
        f_der = dev_activation(x_der)
        t_terme = coefficients[3*2+1] * f_der * X[1][individu]
        somme += delta_prediction * f_der_x * t_terme
    return somme


def der_S_a12(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        t_terme = P_1j(X, coefficients, 0, individu)
        somme += delta_prediction * f_der_x * t_terme
    return somme


def der_S_a22(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        # Troisième facteur
        t_terme = P_1j(X, coefficients, 1, individu)
        somme += delta_prediction * f_der_x * t_terme
    return somme


def der_S_b2(X, Y, coefficients):
    somme = 0
    for individu in range(X.shape[0]):
        # Premier facteur
        N_y_predit = Neural_prediction(X, coefficients, individu)
        delta_prediction = N_y_predit - Y[individu][0]
        # Deuxième facteur
        x_der = coefficients[3*2] * P_1j(X, coefficients, 0, individu)
        x_der += coefficients[3*2+1] * P_1j(X, coefficients, 1, individu)
        x_der -= coefficients[3*2+2]
        f_der_x = dev_activation(x_der)
        somme += delta_prediction * f_der_x
    return -1 * somme



def main(coefficients):
    functions_derivees = [der_S_a111, der_S_a112, der_S_b11, \
                          der_S_a121, der_S_a122, der_S_b12, \
                          der_S_a12, der_S_a22, der_S_b2]
    for i in range(100):
        # Choix de la variable
        indice = random.choice(range(len(coefficients)))
        # Calcul des paramètres à optimiser
        coefficients[indice] += - pas * functions_derivees[indice](X, Y, coefficients)
    return coefficients


coefficients_mat = np.zeros((3, 3))
coefficients_mat_reshape = coefficients_mat.reshape(1,9)[0]
coefficients = main(coefficients_mat_reshape)
print(coefficients)
print(prediction(4, 5, coefficients))