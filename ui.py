from tkinter.constants import HORIZONTAL
from Model import Model
from tkinter import Entry, Tk, Label, Entry, Button, W, E
from tkinter.ttk import Progressbar


def main():
    root = Tk()
    root.title("Comment Sentiment Analysis")
    mymodel = Model()
    mymodel.load_vocab()
    mymodel.load_the_model()
    # print(mymodel.predict("I love watching movie"))
    # print(mymodel.stupid_infer("I love watching movie"))

    label1 = Label(root, width=90)
    label1['text'] = 'Enter your text'
    label1.grid(row=0, column=0, columnspan=3, sticky=W+E, padx=2)

    input = Entry(root, width=70)
    input.grid(row=1, column=0, columnspan=2, sticky=W+E, padx=2)

    def predict_with_model():
        sentence = input.get()
        status_display['text'] = f'Your input text is: '
        displayText['text'] = sentence
        result = mymodel.predict(sentence)
        top1_label['text'] = f"{result[0][0]}: {result[0][1] * 100:.2f}%"
        top2_label['text'] = f"{result[1][0]}: {result[1][1] * 100:.2f}%"
        top3_label['text'] = f"{result[2][0]}: {result[2][1] * 100:.2f}%"

        top1_chart['value'] = result[0][1] * 100
        top2_chart['value'] = result[1][1] * 100
        top3_chart['value'] = result[2][1] * 100

    myButton = Button(root, text="Predict",
                      command=predict_with_model, width=20)
    myButton.grid(row=1, column=2, sticky=E+W, padx=5, pady=5)

    status_display = Label(root, width=90)
    status_display['text'] = 'Waiting for input'
    status_display.grid(row=2, columnspan=3, column=0, sticky=E+W, padx=2)

    displayText = Label(root, width=90)
    displayText.grid(row=3, column=0, columnspan=3, sticky=E+W, padx=2)

    top1_label = Label(root)
    top1_label['text'] = 'Positive'
    top1_label.grid(row=4, column=0, sticky=E+W, padx=2)

    top1_chart = Progressbar(root, orient=HORIZONTAL)
    top1_chart.grid(row=4, column=1, columnspan=2, sticky=W+E, padx=2)

    top2_label = Label(root)
    top2_label['text'] = 'Neutral'
    top2_label.grid(row=5, column=0, sticky=E+W, padx=2)

    top2_chart = Progressbar(root, orient=HORIZONTAL)
    top2_chart.grid(row=5, column=1, columnspan=2, sticky=W+E, padx=2)

    top3_label = Label(root)
    top3_label['text'] = 'Negative'
    top3_label.grid(row=6, column=0, sticky=E+W, padx=2)

    top3_chart = Progressbar(root, orient=HORIZONTAL)
    top3_chart.grid(row=6, column=1, columnspan=2, sticky=W+E, padx=2)

    root.mainloop()


if __name__ == "__main__":
    main()
