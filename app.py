from tensorflow.keras.models import load_model
# Load the tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

with open('Pickle/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the saved model
model = load_model('Models/nep-text-pred.keras')

# Print the model summary to verify it was loaded correctly
# model.summary()
max_length = 8


def generate_seq(model, tokenizer, max_length, seed_text, n_words, top_n=3):
    in_text = seed_text
    generated_words = []
    # Generate a fixed number of words
    for _ in range(n_words):
        # Encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # Pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # Predict probabilities for each word
        yhat = model.predict(encoded, verbose=0)
        # Get top-n indices with highest probabilities
        top_indices = yhat.argsort()[0][-top_n:][::-1]
        # Map predicted word indices to words
        top_words = []
        for idx in top_indices:
            for word, index in tokenizer.word_index.items():
                if index == idx:
                    top_words.append(word)
        # Append top words to generated words list
        generated_words.append(top_words)
    return generated_words

def input_pred():
    while True:
        text = input(   "Enter your line (Enter '0' to exit): ")
        if text == "0":
            print("Execution completed....")
            break
        else:
            try:
                print(generate_seq(model, tokenizer, max_length-1,text,1))
            except Exception as e:
                print("Error occurred:", e)
                continue

def generate_paragraph(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # Generate a fixed number of words
    for _ in range(n_words):
        # Encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # Pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # Predict probabilities for each word
        yhat = model.predict(encoded, verbose=0)
        # Sample a word index from the predicted probabilities
        yhat = np.argmax(yhat)
        # Map predicted word index to word
        out_word = tokenizer.index_word.get(yhat, '')
        # Append to input
        in_text += ' ' + out_word
    return in_text








def switch(choice):
    if choice == 'pre':
        input_pred()
    elif choice == 'para':

        text_seed = str(input("enter seed text"))
        gen_text=generate_paragraph(model, tokenizer, max_length-1,text_seed,80)
        print(gen_text)
        output_file_path = "output/output.txt"
        with open(output_file_path, "w",encoding="utf=8") as file:
            file.write(gen_text + " ")
    else:
        print("Invalid choice. Please enter 'y' or 'n'.")

# Example usage
print("paragraph or prediction???")
user_input = input("Enter 'pre' or 'para': ").strip().lower()
switch(user_input)
