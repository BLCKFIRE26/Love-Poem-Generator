import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


##load and preprocess data
def load_data(data_folder):
    texts = []
    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), 'r', encoding ='utf-8') as file:
            poem = file.read()
            texts.append(poem)

    return texts

data_folder = "data/love_poems"
poems = load_data(data_folder)

#Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poems)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(poems)
input_sequences = [seq[:-1] for seq in sequences]
target_sequences = [seq[1:] for seq in sequences]

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='pre')

#---build the GAN---#

def generator(latent_dim, total_words):
    model = Sequential()
    model.add(Dense(256, input_dim = latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512,))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024,))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(total_words, activation='softmax'))
    return model


def discriminator(max_sequence_length, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

latent_dim = 100
generator = generator(latent_dim, total_words)
discriminator = discriminator(max_sequence_length, total_words)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan = build_gan(generator, discriminator)

#--TRAIN--#

def train_gan(generator, discriminator, gan, input_sequences, target_sequences, epochs=1000, batch_size=200):
    for epoch in range(epochs):
        for batch_start in range(0, len(input_sequences), batch_size):
            batch_end = batch_start + batch_size
            real_poems = target_sequences[batch_start:batch_end]
            
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_poems = generator.predict(noise)
            
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            d_loss_real = discriminator.train_on_batch(real_poems, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_poems, fake_labels)
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            valid_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)
            
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")



def generate_poem(generator, tokenizer, latent_dim, max_sequence_length):
    seed_text = "Love is"
    for _ in range(10):
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        padded_sequence = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
        predicted_word_index = np.argmax(generator.predict(np.random.normal(0, 1, size=[1, latent_dim])))
        predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]
        seed_text += " " + predicted_word
    return seed_text

generator.save_weights("generator_weights.h5")

# Build out the Streamlit App
st.title("Poem Generator")

if st.button("Generate Poem"):
    # Load the saved model weights
    generator.load_weights("generator_weights.h5")

    # Generate and display a poem
    generated_poem = generate_poem(generator, tokenizer, latent_dim, max_sequence_length)
    st.write(generated_poem)


