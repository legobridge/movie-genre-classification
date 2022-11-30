import json
import tkinter as tk
import requests

import pandas as pd

from preprocessing import preprocess_descriptions


def get_probabilities(processed_description: pd.Series):
    # url = 'localhost:42069/small_bert/get_probabilities'
    # params = {'processed_description': processed_description[0]}
    # # response = requests.get(url)
    # with open('../models/genre_encoding.json') as f:
    #     genre_encoding = json.load(f)
    # print(genre_encoding)
    # probability_map = '\n'
    # for
    pass


def classify_input():
    description = inputtxt.get(1.0, "end-1c")
    processed_description = preprocess_descriptions(pd.Series([description]))
    lbl_out.config(text="\nMovie Description:\n\n" + get_probabilities(processed_description))


if __name__ == '__main__':
    frame = tk.Tk()
    frame.title("Movie Genre Classifier")
    frame.geometry('800x500')

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
