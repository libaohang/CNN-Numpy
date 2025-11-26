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
        # convert to grayscale
        img = img.convert("L")
        
        # invert so background is black
        img = Image.eval(img, lambda x: 255 - x)

        # convert to numPy
        img = np.array(img)

        # get bounding box of the drawn digit
        coords = np.where(img < 200)     # find dark pixels
        if coords[0].size == 0:
            return np.zeros((1, 28, 28))  # empty canvas → return blank
        
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()
        img = img[ymin:ymax, xmin:xmax]  # crop to digit

        # resize longest side to 20px (MNIST uses 20×20 digits padded to 28×28)
        h, w = img.shape
        if h > w:
            new_w = int(20 * w / h)
            img = Image.fromarray(img).resize((new_w, 20))
        else:
            new_h = int(20 * h / w)
            img = Image.fromarray(img).resize((20, new_h))

        img = np.array(img)

        # pad to 28×28
        padded = np.zeros((28, 28))
        x_offset = (28 - img.shape[0]) // 2
        y_offset = (28 - img.shape[1]) // 2
        padded[x_offset:x_offset+img.shape[0], y_offset:y_offset+img.shape[1]] = img

        # normalize
        padded = padded.astype("float32") / 255.0

        return padded.reshape(1, 28, 28)

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
        inp = self.preprocess(img)[:, :, : , None]
        pred = predict(self.model, inp).argmax()

        self.label.config(text=f"Prediction: {pred}")
        self.last_x, self.last_y = None, None

    def run(self):
        self.root.mainloop()