import numpy as np
from utils import load_data, train_dev_test, make_map_from_nested, word_vectors, trigram_vectors, all_vectors
from model import simple_embeddings

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

seed = 42
database, top_tokens = load_data('data')

classes = list(set(database['class']))
classes.sort()

train, dev, test = train_dev_test(database, seed)

class_map = {classes[idx]: idx for idx in range(len(classes))}  # class mapping to integers
word_map = make_map_from_nested(top_tokens['top_words'].values())
trigram_map = make_map_from_nested(top_tokens['top_trigrams'].values())

results = set()
for sen in train['text']:
    results.update(sen.split())
all_map = {w: i for i, w in enumerate(list(results))}

y_train_ints = np.array([class_map[label] for label in train['class']])
y_dev_ints = np.array([class_map[label] for label in dev['class']])
y_test_ints = np.array([class_map[label] for label in test['class']])

y_train = to_categorical(y_train_ints, 10)
y_dev = to_categorical(y_dev_ints, 10)
y_test = to_categorical(y_test_ints, 10)

X_train_words = word_vectors(train['text'], word_map)
X_train_trigrams = trigram_vectors(train['text'], trigram_map)
X_train_all = all_vectors(train['text'], all_map)
X_dev_words = word_vectors(dev['text'], word_map)
X_dev_trigrams = trigram_vectors(dev['text'], trigram_map)
X_dev_all = all_vectors(dev['text'], all_map)
X_test_words = word_vectors(test['text'], word_map)
X_test_trigrams = trigram_vectors(test['text'], trigram_map)
X_test_all = all_vectors(test['text'], all_map)

max_word_len = max(max([len(s) for s in X_train_words]),
                   max([len(s) for s in X_dev_words]),
                   max([len(s) for s in X_test_words]))
max_tri_len = max(max([len(s) for s in X_train_trigrams]),
                  max([len(s) for s in X_dev_trigrams]),
                  max([len(s) for s in X_test_trigrams]))
max_all_len = 350

X_train_w = pad_sequences(X_train_words, max_word_len, padding='post')
X_train_t = pad_sequences(X_train_trigrams, max_tri_len, padding='post')
X_train_a = pad_sequences(X_train_all, max_all_len, padding='post')
X_dev_w = pad_sequences(X_dev_words, max_word_len, padding='post')
X_dev_t = pad_sequences(X_dev_trigrams, max_tri_len, padding='post')
X_dev_a = pad_sequences(X_dev_all, max_all_len, padding='post')
X_test_w = pad_sequences(X_test_words, max_word_len, padding='post')
X_test_t = pad_sequences(X_test_trigrams, max_tri_len, padding='post')
X_test_a = pad_sequences(X_test_all, max_all_len, padding='post')

word_vocab_size = len(word_map) + 1  # Adding 1 because of reserved 0 index
word_embedding_size = 200
trigram_vocab_size = len(trigram_map) + 1
trigram_embedding_size = 100
all_vocab_size = len(all_map) + 1
all_embedding_size = 400
dropout_rate = 0.5
batch_size = 512
epochs = 100
num_classes = 10

model = simple_embeddings(word_vocab_size, word_embedding_size, max_word_len,
                          trigram_vocab_size, trigram_embedding_size, max_tri_len,
                          all_vocab_size, all_embedding_size, max_all_len,
                          dropout_rate, num_classes)

earlystop = EarlyStopping(monitor='val_loss', patience=2)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

history = model.fit([X_train_w, X_train_t, X_train_a], y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[earlystop, mc],
                    verbose=1,
                    validation_data=([X_dev_w, X_dev_t, X_dev_a], y_dev))
