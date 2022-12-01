import json
import tkinter as tk

import numpy as np
import pandas as pd
import requests

from preprocessing import preprocess_descriptions


# ssh -N -L 42069:batgirl.eecs.northwestern.edu:5000 kah3465@batgirl.eecs.northwestern.edu
def get_top_genre(genre_encoding, response):
    top = '0'
    mx = response['0']
    for key, value in response.items():
        if value > mx:
            top = key
            mx = value
    return genre_encoding[top]


def get_probabilities(processed_description: pd.Series):
    url = 'http://127.0.0.1:42069/small_bert/get_probabilities'
    params = {'processed_description': processed_description[0]}
    response = requests.get(url, params).json()['probabilities']
    with open('../models/genre_encoding.json') as f:
        genre_encoding = json.load(f)
    probability_map = f'\nTop prediction: {get_top_genre(genre_encoding, response)}\n\n'
    for key, value in response.items():
        genre = genre_encoding[key]
        probability = np.round(value * 100, 2)
        probability_map += genre + ' : ' + str(probability) + '%\n'
    return probability_map


def classify_input():
    description = inputtxt.get(1.0, "end-1c")
    processed_description = preprocess_descriptions(pd.Series([description]))
    lbl_out.config(text="\nPredicted Probabilities:\n" + get_probabilities(processed_description))


if __name__ == '__main__':
    frame = tk.Tk()
    frame.title("Movie Genre Classifier")
    frame.geometry('1000x700')

    lbl = tk.Label(frame, text="Enter Movie Description Below")
    lbl.pack()

    inputtxt = tk.Text(frame,
                       height=20,
                       width=90)

    inputtxt.pack()

    printButton = tk.Button(frame,
                            text="Classify",
                            command=classify_input)
    printButton.pack()

    lbl_out = tk.Label(frame, text="")
    lbl_out.pack()

    frame.mainloop()
