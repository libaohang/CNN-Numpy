import tkinter as tk
import numpy as np
from PIL import Image, ImageGrab

class DigitGUI:

    def __init__(self, model):
        self.model = model
        
        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        tk.Button(self.root, text="Predict", command=self.predict_digit).pack()
        self.label = tk.Label(self.root, text="Draw a digit", font=("Arial", 16))
        self.label.pack()

        def clear_canvas():
            self.canvas.delete("all")

        clear_button = tk.Button(self.root, text="Clear", command=clear_canvas)
        clear_button.pack()

        self.last_x, self.last_y = None, None

    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=15)
        self.last_x, self.last_y = event.x, event.y

    def preprocess(self, img):
        img = img.resize((28, 28)).convert("L")
        img = np.array(img) / 255.0
        return img.reshape(1, 28, 28)

    def predict_digit(self):
        def predict(network, images):
            for layer in network:
                images = layer.forward(images)
            return images

        # Get canvas position & grab image
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x2 = x + self.canvas.winfo_width()
        y2 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x2, y2))

        # Preprocess and predict
        inp = self.preprocess(img)
        pred = predict(self.model, inp).argmax()

        self.label.config(text=f"Prediction: {pred}")
        self.last_x, self.last_y = None, None

    def run(self):
        self.root.mainloop()