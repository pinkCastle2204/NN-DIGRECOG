from PIL import Image, ImageDraw, ImageTk
import numpy as np
from nnfs.layers import Dense
from nnfs.activations import relu, softmax
import tkinter as tk

SCALE       = 10
IMG_SIZE    = 28
CANVAS_SIZE = IMG_SIZE * SCALE   # 280

layer1 = Dense(784, 128)
layer2 = Dense(128, 10)
layer1.W = np.load('W1.npy')
layer1.b = np.load('b1.npy')
layer2.W = np.load('W2.npy')
layer2.b = np.load('b2.npy')

pil_image = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
draw_pil  = ImageDraw.Draw(pil_image)

def refresh_canvas():
    zoomed   = pil_image.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)
    tk_image = ImageTk.PhotoImage(zoomed)
    canvas.create_image(0, 0, anchor='nw', image=tk_image)
    canvas.tk_image = tk_image   # prevent garbage collection

def predict():
    img_array = np.array(pil_image) / 255.0
    img_flat  = img_array.reshape(1, 784)

    Z1 = layer1.forward(img_flat)
    A1 = relu(Z1)
    Z2 = layer2.forward(A1)
    A2 = softmax(Z2)

    prediction = np.argmax(A2)
    confidence = np.max(A2) * 100
    print(f"Predicted: {prediction} | Confidence: {confidence:.2f}%")

last_x = None
last_y = None

def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y

def draw(event):
    global last_x, last_y
    x = event.x // SCALE
    y = event.y // SCALE
    draw_pil.rectangle([x, y, x+1, y+1], fill=255)   # draws on 28x28 image
    refresh_canvas()                    # updates display
    last_x = event.x
    last_y = event.y

def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None

def clear_canvas():
    draw_pil.rectangle([0, 0, IMG_SIZE, IMG_SIZE], fill=0)
    refresh_canvas()

root = tk.Tk()
root.title("NNFS-Digit-Recog")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                   bg='black', cursor='cross')
canvas.pack(padx=10, pady=10)

canvas.bind('<Button-1>',        start_draw)
canvas.bind('<B1-Motion>',       draw)
canvas.bind('<ButtonRelease-1>', stop_draw)

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

tk.Button(button_frame, text='Predict', width=12, command=predict).pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text='Clear',   width=12, command=clear_canvas).pack(side=tk.LEFT, padx=5)

root.mainloop()