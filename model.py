from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, Flatten, Concatenate,\
    Conv1D, MaxPooling1D, BatchNormalization


def simple_embeddings(word_vocab_size,
                      word_embedding_size,
                      word_input_length,
                      trigram_vocab_size,
                      trigram_embedding_size,
                      trigram_input_length,
                      all_vocab_size,
                      all_embedding_size,
                      all_input_length,
                      dropout_rate,
                      num_classes):
    word_input = Input(shape=(word_input_length,))
    trigram_input = Input(shape=(trigram_input_length,))
    all_input = Input(shape=(all_input_length,))

    word_embedding = Embedding(input_dim=word_vocab_size,
                               output_dim=word_embedding_size,
                               input_length=word_input_length,
                               trainable=True)(word_input)
    x1 = Conv1D(26, 5, activation='relu')(word_embedding)
    x1 = MaxPooling1D(2)(x1)
    x1 = Flatten()(x1)
    x1 = BatchNormalization(trainable=True)(x1)
    x1 = Dropout(dropout_rate)(x1)
    x1 = Model(inputs=word_input, outputs=x1)

    trigram_embedding = Embedding(input_dim=trigram_vocab_size,
                               output_dim=trigram_embedding_size,
                               input_length=trigram_input_length,
                               trainable=True)(trigram_input)
    x2 = Conv1D(26, 5, activation='relu')(trigram_embedding)
    x2 = MaxPooling1D(2)(x2)
    x2 = Flatten()(x2)
    x2 = BatchNormalization(trainable=True)(x2)
    x2 = Dropout(dropout_rate)(x2)
    x2 = Model(inputs=trigram_input, outputs=x2)

    all_embedding = Embedding(input_dim=all_vocab_size,
                              output_dim=all_embedding_size,
                              input_length=all_input_length,
                              trainable=True)(all_input)
    x3 = Conv1D(52, 10, activation='relu')(all_embedding)
    x3 = MaxPooling1D(4)(x3)
    x3 = Flatten()(x3)
    x3 = BatchNormalization(trainable=True)(x3)
    x3 = Dropout(dropout_rate)(x3)
    x3 = Model(inputs=all_input, outputs=x3)

    combined = Concatenate()([x1.output, x2.output, x3.output])
    z = Dense(512, activation='relu')(combined)
    z = BatchNormalization(trainable=True)(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(256, activation='relu')(z)
    z = BatchNormalization(trainable=True)(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[x1.input, x2.input, x3.input], outputs=z)

    return model