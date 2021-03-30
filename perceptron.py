import numpy as np


X_1 = [i/20 for i in range(1, 21)]
X_2 = [2*(i-2)/20 if i%2==0 else 2*(i+2)/20 for i in range(1, 21)]
X = np.array([X_1, X_2])
Y = np.array([0 if i%2==0 else 1 for i in range(1, 21)])


# np.random.seed(42)

W = np.random.randn(1, 2)
biais = np.random.randn(1, 1)

epoch = 100
lr = 0.1


def activation(W, X, B):
    Y_predit = np.dot(W, X) - B
    return [1 if y_i > 0 else 0 for y_i in Y_predit[0]]


def erreur(Y_predit, Y_vrai):
    return abs(Y_predit-Y_vrai).sum()/len(Y_predit)


erreur_predition = 1
iteration = 0
while erreur_predition > 0.01 and iteration < epoch : # Taux d'erreur de 10%
    iteration += 1
    for individu in range(X.shape[1]):
        Y_predit = activation(W, X, biais)
        for i in range(X.shape[0]):
            W[0][i] += lr * (Y[individu] - Y_predit[individu]) * X[i][individu]
        biais += -lr * (Y[individu] - Y_predit[individu])
    Y_predit = activation(W, X, biais)
    erreur_predition = erreur(Y_predit, Y)
    if iteration%10==0:
        print("iteration ", str(iteration), " erreur : ", str(erreur_predition))

print("Poids et biais :")
print(W, biais)
print("Prédiction : ")
print(activation(W, X, biais))
print("Itération : ", str(iteration))