import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

from sklearn.model_selection import GridSearchCV

# Loading training data

with open('TrainingVectorFeatures.pkl', 'rb') as trainfile:
    features = pickle.load(trainfile)

with open('TrainingLabels.txt', 'r') as infile2:
    labels = [line.strip() for line in infile2]

# Data preprocessing

labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(labels)
y_train_one_hot = to_categorical(y_train)

X_train = np.array(features) #adding "step" dimensions


def create_model(dropout=0.0):
    model = Sequential()

    model.add(Conv1D(13, kernel_size=(10),
                     input_shape=(340, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling1D(2, padding='same'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

#load data
X = X_train
Y = y_train_one_hot

#create model
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=512, verbose=0)

#define the grid search parameters
dropout = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

num_classes = 13
batch_size = 512

#compile
param_grid = dict(dropout=dropout)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=10)
grid_result = grid.fit(X,Y)
#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
