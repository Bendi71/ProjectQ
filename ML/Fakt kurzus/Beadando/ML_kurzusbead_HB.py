"""
This code is my solution to the assignment for the course "Machine Learning" at FAKT Szakkollégium.
The assigment was a Kaggle competition, where the goal was to predict the popularity of articles based on the given
data. I added the data to the repository, so you can check it out.

Description of the task: This is a private binary classification competition for the Introduction to Machine Learning
Tools, at fakt in the Fall semester of 2022/23.

In this competition, your task is to predict which articles are shared the most on social media. The data comes from the
 website mashable.com from the beginning of 2015. The dataset used in the competition can be found in the UCI repository.

There is a public leaderboard that is computed on about 60% of the test data and a private one that is computed on the
remaining observations. You can use the public leaderboard to evaluate your current standing till the deadline date.
The final rankings will be based on the private leaderboard. Bear in mind that by submitting many times to the public
leaderboard and making decisions upon these results you might overfit your model.
"""

'''A feladat teljesítéséhez egy többrétegű perceptron modellt használtam. Ehhez kiinduló pontként a következő Jeff 
Heaton által készített modellt vettem. https://github.com/jeffheaton/t81_558_deep_learning/blob/master
/t81_558_class_04_2_multi_class.ipynb Az adatok megtisztításával kezdtem, kivettem az articel_id, illetve a timedelta 
oszlopokat, melyek nem alkalmasak az eredményváltozó magyarázásra, illetve standardizáltam a nem binomiális 
változókat, hogy azonos skálán mozogjanak. A rétegek közötti aktivációs függvény a Relu, míg az utolsó rétegben a 
sigmoid, erre azért van szükség, hogy a kimeneti érték a (0,1) intervallumban legyen Zhang et al. (2023). A modell a 
túlilleszkedés megelőzésére végig az Early stopping technikát alkalmazta, mely lényegége, hogy ha nem érünk el egy 
bizonyos javulást a validációs halmazon, akkor befejeződik az optimalizáció Zhang et al. (2023). Kipróbáltam, 
hogy a Ridge büntető paraméter használatával más eredményt kapunk e, de nem javult tőle a modellünk, viszont a futási 
idő jelentősen megugrott. A forrásul használt modellben az optimalizációs módszer az Adam volt, azonban a 
Sztochasztikusan ereszkedő gradiens módszerrel 0.01 tanulási rátával jobb eredményt lehet elérni, így a végleges 
modellben ezt a módszert alkalmaztam. Az eredeti modellben szereplő rétegekhez hozzáadtam plusz egyet, mivel ez is 
javította a modell hatékonyságát, viszont tovább bővítve azt már nem változtatott a kimenet eredményességén. A 
hiperparaméterek változtatásával próbáltam még tovább fejleszteni a  modellt, egyrészt az Early stopping-ban lévő 
minimum elvárt változás mértékének változtatásával, viszont a kezdeti 1e-3-as érték bizonyult a legjobbnak. Ezt 
követően a SGD-ben lévő tanulási rátát próbáltam meg optimalizálni. Egyrészt megpróbáltam dinamikusan változó 
learning rate-t használni, de ez nem járt különösebb eredménnyel. Ehhez a ChatGpt által ajánlott módszert használtam. 
Majd megvizsgáltam, hogy állandó tanulási ráta esetén mi a legjobb. Ezt először egy 0.01 tartalmazó szélesebb sávban 
vizsgáltam meg, majd miután azt láttam, hogy a 0.01-hez közeli értékek az ideálisak, ezért egy szűkített 
intervallumot is megnéztem. Amelyből egyébként az jött ki, hogy a 0.011 a legjobb, viszont a teszt adatbázison ez 
valamelyest rosszabbul teljesített, ezért maradtam a 0.01-nél. Végezetül, mivel a train adatbázisban a 0-ás 
popularitású változóból 26116 db van, míg az 1-es popularitásúból 3617 db van, emiatt a SMOTE algoritmus segítségével 
megpróbáltam több egyes polaritású változót generálni, de ez negatívan hatott a végső kimenetre. Ennek a működéséről 
itt olvastam: https://www.geeksforgeeks.org/introduction-to-resampling-methods/.

Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). Dive into deep learning. Cambridge University Press.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

df = pd.read_csv('train.csv')

print(df)
df = df.drop('article_id', axis=1).drop('timedelta', axis=1)
columns_to_standardize = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words',
                          'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
                          'average_token_length', 'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
                          'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
                          'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess',
                          'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
                          'global_sentiment_polarity', 'global_rate_positive_words', 'global_rate_negative_words',
                          'rate_positive_words', 'rate_negative_words', 'avg_positive_polarity',
                          'min_positive_polarity', 'max_positive_polarity', 'avg_negative_polarity',
                          'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
                          'title_sentiment_polarity', 'abs_title_subjectivity', 'abs_title_sentiment_polarity']
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(x)
print(y)

'''
#Resampling módszer
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x, y)
'''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

'''
#Egy réteggel bővített neurális háló, Adam optimalizálás.
model = Sequential()

model.add(Dense(200, input_dim=x.shape[1], activation='relu',
                kernel_initializer='random_normal'))
model.add(Dense(100, activation='relu',
                kernel_initializer='random_normal'))
model.add(Dense(50,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(1,activation='sigmoid',kernel_initializer='random_normal'))
model.compile(
    loss='binary_crossentropy', 
    optimizer=tensorflow.keras.optimizers.Adam(),
    metrics=[AUC(name='auc'), 'accuracy']
monitor = EarlyStopping(monitor='val_auc', min_delta=1e-3, 
 patience=5, verbose=1, mode='auto', restore_best_weights=True)
)
'''

'''
#Ridge büntető paraméter
model = Sequential()

model.add(Dense(100, activation='relu',input_dim=x.shape[1], kernel_initializer='random_normal', 
    kernel_regularizer=l2(0.01)))
model.add(Dense(50, activation='relu', kernel_initializer='random_normal', kernel_regularizer=l2(0.01)))
model.add(Dense(25, activation='relu', kernel_initializer='random_normal', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(), metrics=[AUC(name='auc'), 'accuracy'])


history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=64, verbose=2)
'''

'''
#Early stopping hiperparaméter tuning
min_delta_values = [1e-5, 1e-4, 1e-3, 1e-2]

for min_delta in min_delta_values:
    monitor = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=5, verbose=1, mode='auto', restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=1000)
    test_predictions = model.predict(x_test)
    test_auc = roc_auc_score(y_test, test_predictions)

    print(f'Min Delta: {min_delta}, Test AUC: {test_auc}')
'''

# Végeleges modell SGD optimalizálással.
model = Sequential()

model.add(Dense(200, input_dim=x.shape[1], activation='relu',
                kernel_initializer='random_normal'))
model.add(Dense(100, activation='relu',
                kernel_initializer='random_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(25, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(lr=0.01),
    metrics=[AUC(name='auc'), 'accuracy']
)
monitor = EarlyStopping(monitor='val_auc', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto', restore_best_weights=True)

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          callbacks=[monitor], verbose=2, epochs=1000)

'''
#Dinamikusan változtatott tanulási ráta
def lr_schedule(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return float(lr)

r_scheduler = LearningRateScheduler(lr_schedule)

monitor = EarlyStopping(monitor='val_auc', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                    epochs=1000, batch_size=64, verbose=2, callbacks=[lr_scheduler, monitor])
'''

'''
#Állandó tanulási ráta széles intervallum
learning_rates = [0.001,0.01,0.02,0.05,0.01]
train_auc_scores = []

for lr in learning_rates:
    model = create_model(lr)

    monitor = EarlyStopping(monitor='val_auc', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        callbacks=[monitor], verbose=2, epochs=1000)

    test_predictions = model.predict(x_test)
    test_auc = roc_auc_score(y_test, test_predictions)
    train_auc_scores.append(test_auc)


plt.plot(learning_rates, train_auc_scores, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Test AUC Score')
plt.title('Learning Rate Hyperparameter Tuning')
plt.show()
'''

'''
#Állandó tanulási ráta szűk intervallum
learning_rates = [0.0099,0.01,0.011,0.0115,0.012]
train_auc_scores = []

for lr in learning_rates:
    model = create_model(lr)

    monitor = EarlyStopping(monitor='val_auc', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        callbacks=[monitor], verbose=2, epochs=1000)

    test_predictions = model.predict(x_test)
    test_auc = roc_auc_score(y_test, test_predictions)
    train_auc_scores.append(test_auc)


plt.plot(learning_rates, train_auc_scores, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Test AUC Score')
plt.title('Learning Rate Hyperparameter Tuning')
plt.show()
'''
# Eredmény kiszámítása és kiírása, result.csv-ben egy oszlop van, melyben a test halmazban szereplő változók article_id-jei szerepelnek.
df = pd.read_csv('test.csv')
df = df.drop('article_id', axis=1).drop('timedelta', axis=1)

columns_to_standardize = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words',
                          'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
                          'average_token_length', 'num_keywords', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
                          'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
                          'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess',
                          'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
                          'global_sentiment_polarity', 'global_rate_positive_words', 'global_rate_negative_words',
                          'rate_positive_words', 'rate_negative_words', 'avg_positive_polarity',
                          'min_positive_polarity', 'max_positive_polarity', 'avg_negative_polarity',
                          'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
                          'title_sentiment_polarity', 'abs_title_subjectivity', 'abs_title_sentiment_polarity']

scaler = StandardScaler()

df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

x = df
pred = model.predict(x)

rf = pd.read_csv('result.csv')
rf['score'] = pred
rf.to_csv('result.csv', index=False)

'''
test_predictions = model.predict(x_test)

test_auc = roc_auc_score(y_test, test_predictions)

print(test_auc)
'''
