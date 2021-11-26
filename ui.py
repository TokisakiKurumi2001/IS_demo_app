from Model import Model
import tkinter
    
def main():
    root = tkinter.Tk()
    mymodel = Model()
    mymodel.load_vocab()
    mymodel.load_the_model()
    # print(mymodel.predict("I love watching movie"))
    # print(mymodel.stupid_infer("I love watching movie"))

    e = tkinter.Entry(root, width=50)
    e.pack()

    def predict_with_model():
        sentence = e.get()
        status_display['text'] = f'Your sentence: \'{sentence}\' is labeled as: '
        # text_result['text'] = mymodel.stupid_infer(sentence)
        text_result['text'] = mymodel.predict(sentence)

    myButton = tkinter.Button(root, text="Predict", command=predict_with_model)
    myButton.pack()

    status_display = tkinter.Label(root)
    status_display['text'] = 'Waiting for input'
    status_display.pack()

    text_result = tkinter.Label(root)
    text_result['text'] = ''
    text_result.pack()

    root.mainloop()

if __name__ == "__main__":
    main()