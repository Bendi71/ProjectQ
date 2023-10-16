'''
Ne kerüljenek bele olyan beták, amik nem szignifikánsak
Ha a lambda magas, akkor a beta=0 és a tengelymetszet az átlag lesz
LASSO: SSR + penalty for complex models
                lambda * sum(abs(beta_i))
RIDGE: SSR + lambda * sum(beta_i^2)
    Akkor használjuk, ha kevesebb megfigyelésünk van, mint változónk
    Az invertálási problémát megoldja a béta négyzetelésével

elastic net: SSR + lambda * sum(abs(beta_i)) + lambda * sum(beta_i^2) - lasso és ridge komibnációja

A lambda egy hiperparaméter, amit a cross validationnal kell meghatározni
'''

"""
y=x1+x2 uniform(0,1) - ezt fogjuk LASSO-val megbecsülni
n=20
eps ~ N(0,2)
X0=(0,0)


1. data generálás
2. LASSO becslés
3. predict for (0,0)
4. run this 1000 times
5. bias, var, mse
6. run with different lambdas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

M = 100
preds = np.full(shape=(M, 100), fill_value=0.0)
n_coef = np.full(shape=(M, 100), fill_value=0.0)
lam = np.linspace(0.01, 0.5, 100)
for i in range(M):
    n = 20
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    eps = np.random.normal(0, 2, n)
    y = x1 + x2 + eps
    for ida, a in enumerate(lam):
        model = Lasso(alpha=a)
        model.fit(np.array([x1, x2]).T, y)
        preds[i,ida] = model.predict(np.array([0, 0]).reshape(1, -1))
        n_coef[i,ida] = np.sum(model.coef_ != 0)

bias = np.mean(preds, axis=0)
var = np.var(preds, axis=0)
mse = bias ** 2 + var



"""
Határozzuk meg a megfelelő lambda értéket keresztvalódással
Ehhez kell a LAsoCV függvény
"""

from sklearn.linear_model import LassoCV
n = 10000
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)
eps = np.random.normal(0, 2, n)
y = x1 + x2 + eps
lasso_cv=LassoCV(cv=5, alphas=lam).fit(np.array([x1, x2]).T, y)
print(lasso_cv.alpha_)

"""
Plotoljuk a bias, var, mse-t a lambda függvényében
A bias a lambda növekedésével nő, a variancia csökken
"""
plt.plot(lam, bias**2, label='bias')
plt.plot(lam, var, label='var')
plt.plot(lam, mse, label='mse')
plt.plot(lam, np.mean(n_coef,axis=0), label='n_coef')
plt.legend()
plt.show()
