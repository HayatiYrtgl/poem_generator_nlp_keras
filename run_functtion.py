from keras.models import load_model
from keras.utils import pad_sequences
import joblib
import numpy as np

model = load_model("models/poem")
tok = joblib.load("../tokenizers/tokenizer_sonnet.pkl")


# generate text function
def generate_text(model, tokenizer, seq_len=19, seed_text=None, num_generator=80):

    output = []
    input_text = seed_text

    output.append(input_text)

    for i in range(num_generator):
        encoded = tokenizer.texts_to_sequences([input_text])[0]  # Notice the [0]
        pad_encode = pad_sequences([encoded], maxlen=seq_len, truncating="pre")  # Pass [encoded] as a list

        pred_probabilities = model.predict(pad_encode, verbose=0)[0]
        pred_word_index = np.argmax(pred_probabilities)
        pred_word = tokenizer.index_word[pred_word_index]

        input_text += " " + pred_word

        output.append(pred_word)
    return " ".join(output)

print(generate_text(model=model, tokenizer=tok, seq_len=19, num_generator=80, seed_text="What is that"))
