import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Olvassuk be a FAANG részvények adatait, amik csv fájlban vannak tárolva.
Először állítsuk be, hogy melyik mappából olvassuk be a fájlokat.
"""
mappa= 'C:/Users/Pinter Andrea/Desktop/'

Meta= pd.read_csv(mappa+'META.csv')
print(type(Meta))
print(type(Meta['Date']))
Meta['Date']= pd.to_datetime(Meta['Date'])
print(Meta.shape, Meta.info())
print(Meta.describe())
Meta= Meta.set_index('Date')

Apple= pd.read_csv(mappa+'AAPL.csv', index_col=0, )
Amazon= pd.read_csv(mappa+'AMZN.csv', index_col=0)
Nvidia= pd.read_csv(mappa+'NVDA.csv', index_col=0)
Google= pd.read_csv(mappa+'GOOG.csv', index_col=0)


"""
Nézzük meg, hogy milyen típusú objektumokat kaptunk.
Nézzük meg, hogy milyen oszlopok vannak.
"""

pd.set_option('display.max_columns', None)
print(Meta.head())

"""
Számoljuk ki az Apple napi tartományát (high-low)
Keressük meg azokat a napokat, amikor a tartomány 5nél nagyobb volt.
"""

Apple['Range']= Apple['High']-Apple['Low']
print(Apple['Range'])
print(Apple.loc[Apple['Range']>5])

"""
Rajzoljuk ki a záróárfolyamokat.
"""

plt.plot(Meta['Close'])
plt.title('Meta záróárfolyama')
plt.show()


"""
Fűzzük össze a záróárakat egy táblázatba.
"""

zaro= pd.concat([Apple['Close'], Amazon['Close'], Nvidia['Close'], Google['Close']], axis=1)
zaro.columns= ['Apple', 'Amazon', 'Nvidia', 'Google']
print(zaro)

plt.plot(zaro)
plt.title('Záróárfolyamok')
plt.legend(zaro.columns)
plt.show()

"""
Nézzük meg a napi hozamokat (pct_change).
Számoljuk ki a napi hozamok átlagát, szórását és korrelációját.
Ábrázoljuk az Apple napi hozamának eloszlását piros színnel.
"""
hozamok= zaro.pct_change().dropna()

#logreturns

logreturns=

print(np.log(zaro).diff().dropna())
print(hozamok)

mean= np.mean(hozamok, axis=0)
#print(mean)

std= np.std(hozamok, axis=0)
#print(std)

corr= hozamok.corr()
#print(corr)

plt.hist(hozamok['Apple'], bins=50,color='red')
plt.title('Apple napi hozamának eloszlása')
#plt.show()

"""
Ábrázoljuk az Apple napi hozamát box ploton
Ábrázoljuk az Apple és Amazon napi hozamának kapcsolatát.
"""

plt.boxplot(hozamok['Apple'])
plt.title('Apple napi hozamának box plotja')
#plt.show()

plt.scatter(hozamok['Apple'], hozamok['Amazon'])
plt.title('Apple és Amazon napi hozamának kapcsolata')
#plt.show()

"""
A feladat:
Alkossunk egy portfoliót a 4 részvényből, a súlyokat egyedileg válasszuk meg.
Számoljuk ki:
1) a portfolió napi hozamait
2) a portfolió hozamainak szórását
3) a portfolió hozamának átlagát
4) a max drawdown-t
5) a portfolió hozamát
"""

sulyok= np.array([0.2, 0.3, 0.1, 0.4])
porthozamok_napi= np.dot(hozamok, sulyok)
portreturn= np.sum(porthozamok_napi)
stdev = np.sqrt(np.dot(sulyok.T, np.dot(hozamok.cov(), sulyok)))
portreturns = porthozamok_napi.mean()
maxdraw = porthozamok_napi.min()

print('A portfolió hozama: ', portreturn)
print('A portfolió hozamainak szórása: ', stdev)
print('A portfolió hozamainak átlaga: ', portreturns)
print('A portfolió max drawdown-ja: ', maxdraw)
