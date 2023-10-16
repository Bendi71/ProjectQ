import sklearn.linear_model
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(20230918)
#üres elteres lista
elteres = []
#egy példa itt: generalok 10000 elemet, amire raillesztem a regressziot
X = np.random.uniform(0,1,10000)
eps = np.random.normal(0,1,10000)
y = X**2 -1.5*X + eps
X=X.reshape(-1,1)
reg = sklearn.linear_model.LinearRegression().fit(X, y)
m = np.array(0)
m = m.reshape(-1,1)
elteres.append(reg.predict(m))
elteres=[]
N=1000
M=10000 #ez minel nagyobb, annal jobban teljesit a kvadratikus
for i in range(N):
    X = np.random.uniform(0,1,M)
    eps = np.random.normal(0,1,M)
    y = X**2 -1.5*X + eps
    X=X.reshape(-1,1) #ez kell valamiert a fuggvenyhez
    reg = sklearn.linear_model.LinearRegression().fit(X, y) #megcsinalja a modellt
    m = np.array(0) #az x = 0 pontban
    m = m.reshape(-1,1)
    elteres.append(reg.predict(m)) #hozzaadja az N db elterest, az elteres itt maga a prediktalt ertek, hiszen f(0)=0
bias = np.mean(elteres) #bias: elteresnegyzet atlaga
print(bias)
variance = np.var(elteres)
print(variance)
mse = bias**2 + variance #mse ketfelekepp szamolhato
print(mse)
sklearn.metrics.mean_squared_error(elteres, np.zeros(N))
#ugyanez a kvadratikus linreggel
elteres=[]
N=1000
M = 10000
for i in range(N):
    X = np.random.uniform(0,1,M)
    eps = np.random.normal(0,1,M)
    y = X**2 -1.5*X + eps
    X=X.reshape(-1,1)
    poly_features = PolynomialFeatures(degree=2, include_bias=False) #itt a degree=2 veszi be x^2-et
    X_poly = poly_features.fit_transform(X) #ez kell mint az elobb a reshape
    reg2 = sklearn.linear_model.LinearRegression().fit(X_poly, y)
    m = np.array([[0]])
    m_poly = poly_features.transform(m)
    elteres.append(reg2.predict(m_poly))
bias = np.mean(elteres)
print(bias)
variance = np.var(elteres)
print(variance)
mse = bias**2 + variance
print(mse)
sklearn.metrics.mean_squared_error(elteres, np.zeros(N))
#Nagy M eseten a kvadratikus MSE-je kisebb, mert kicsi a bias, es a nagy elemszam miatt nem sokkal nagyobb a variancia.