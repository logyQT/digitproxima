import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# 1. Load digits
# -----------------------------
digits = load_digits()
X = digits.data
y = digits.target
images_8x8 = digits.images

# -----------------------------
# 2. Preprocessing
# -----------------------------
def preprocess_to_32(img):
    img32 = img.resize((32, 32), Image.Resampling.LANCZOS)
    img32 = ImageOps.invert(img32)
    return img32

def preprocess_to_8x8(img32):
    arr32 = np.array(img32)
    arr32 = (arr32 > 50).astype(int)

    arr8 = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            block = arr32[i*4:(i+1)*4, j*4:(j+1)*4]
            arr8[i, j] = block.sum()
    return arr8

# -----------------------------
# 3. Tkinter window
# -----------------------------
root = tk.Tk()
root.title("KNN OCR ")

root.option_add("*Font", ("Arial", 18))

main = ttk.Frame(root)
main.pack(padx=20, pady=20)

# -----------------------------
# LEWA KOLUMNA
# -----------------------------
left = ttk.Frame(main)
left.grid(row=0, column=0, padx=20)

canvas_size = 300
canvas = tk.Canvas(left, width=canvas_size, height=canvas_size, bg="white")
canvas.pack()

image = Image.new("L", (canvas_size, canvas_size), color=255)
draw = ImageDraw.Draw(image)

last_x, last_y = None, None

def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw_line(event):
    global last_x, last_y
    canvas.create_line(last_x, last_y, event.x, event.y,
                       width=18, fill="black", capstyle=tk.ROUND)
    draw.line([last_x, last_y, event.x, event.y], fill=0, width=18)
    last_x, last_y = event.x, event.y

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_line)

# -----------------------------
# POD PREVIEW – PRZYCISKI I K
# -----------------------------
controls = ttk.Frame(left)
controls.pack(pady=20)

ttk.Label(controls, text="k value:").grid(row=0, column=0, sticky="e")
k_entry = ttk.Entry(controls, width=5)
k_entry.insert(0, "41")
k_entry.grid(row=0, column=1, sticky="w")

result_label = ttk.Label(controls, text="Prediction: ?")
result_label.grid(row=1, column=0, columnspan=2, pady=20)

ttk.Button(controls, text="Clear", command=lambda: clear_canvas()).grid(row=2, column=0, pady=20, ipadx=20, ipady=10)
ttk.Button(controls, text="Predict", command=lambda: predict()).grid(row=2, column=1, pady=20, ipadx=20, ipady=10)

# -----------------------------
# PRAWA KOLUMNA – 2 heatmapy + preview 32×32
# -----------------------------
right = ttk.Frame(main)
right.grid(row=0, column=1, padx=20, sticky="n")

titles = ["Average of k neighbors", "Composite diff", "32×32 preview"]

fig_h = []
ax_h = []
canvas_h = []

for i in range(3):
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    fig_h.append(fig)
    ax_h.append(ax)
    c = FigureCanvasTkAgg(fig, master=right)
    c.get_tk_widget().pack(pady=15)
    canvas_h.append(c)

def update_heatmap(idx, data):
    ax_h[idx].clear()
    ax_h[idx].imshow(data, cmap="hot")
    ax_h[idx].set_title(titles[idx])
    ax_h[idx].set_xticks([])
    ax_h[idx].set_yticks([])
    canvas_h[idx].draw()

def update_preview32(img32):
    ax_h[2].clear()
    ax_h[2].imshow(img32, cmap="gray")
    ax_h[2].set_title("32×32 preview")
    ax_h[2].set_xticks([])
    ax_h[2].set_yticks([])
    canvas_h[2].draw()

# -----------------------------
# WYKRES GŁOSOWANIA
# -----------------------------
vote_col = ttk.Frame(main)
vote_col.grid(row=0, column=2, padx=20, sticky="ns")

fig_vote, ax_vote = plt.subplots(figsize=(3.5, 10))
canvas_vote = FigureCanvasTkAgg(fig_vote, master=vote_col)
canvas_vote.get_tk_widget().pack(expand=True, fill="both", pady=10)

def update_vote_chart(votes):
    ax_vote.clear()
    ax_vote.bar(range(10), votes, color="orange")
    ax_vote.set_title("k-NN voting", fontsize=20)
    ax_vote.set_xticks(range(10))
    ax_vote.set_yticks([])
    ax_vote.set_ylim(0, max(votes) + 1)
    canvas_vote.draw()

# -----------------------------
# CLEAR
# -----------------------------
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=255)
    update_preview32(np.zeros((32, 32)))
    update_heatmap(0, np.zeros((32, 32)))
    update_heatmap(1, np.zeros((32, 32)))
    result_label.config(text="Prediction: ?")

# -----------------------------
# PREDICT
# -----------------------------
def composite_neighbors_diff(img_flat, img32, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan", algorithm="brute")
    knn.fit(X, y)

    dists, idxs = knn.kneighbors(img_flat, n_neighbors=k)
    idxs = idxs[0]

    acc = np.zeros((32, 32), dtype=float)

    for idx in idxs:
        neighbor_img8 = images_8x8[idx]
        neighbor32 = Image.fromarray((neighbor_img8 / 16 * 255).astype(np.uint8)).resize((32, 32))
        acc += np.array(neighbor32, dtype=float)

    avg_neighbor = acc / k
    diff = np.abs(avg_neighbor - np.array(img32, dtype=float))
    return diff

def average_neighbors(img_flat, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan", algorithm="brute")
    knn.fit(X, y)

    dists, idxs = knn.kneighbors(img_flat, n_neighbors=k)
    idxs = idxs[0]

    acc = np.zeros((32, 32), dtype=float)

    for idx in idxs:
        neighbor_img8 = images_8x8[idx]
        neighbor32 = Image.fromarray((neighbor_img8 / 16 * 255).astype(np.uint8)).resize((32, 32))
        acc += np.array(neighbor32, dtype=float)

    return acc / k

def voting_visualization(img_flat, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan", algorithm="brute")
    knn.fit(X, y)

    dists, idxs = knn.kneighbors(img_flat, n_neighbors=k)
    idxs = idxs[0]

    votes = np.zeros(10, dtype=int)
    for idx in idxs:
        votes[y[idx]] += 1

    return votes

def predict():
    img = image.copy()
    img32 = preprocess_to_32(img)
    arr8 = preprocess_to_8x8(img32)
    img_flat = arr8.flatten().reshape(1, -1)

    update_preview32(img32)

    try:
        k = int(k_entry.get())
    except:
        result_label.config(text="Invalid k")
        return

    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan", algorithm="brute")
    knn.fit(X, y)
    pred = knn.predict(img_flat)[0]
    result_label.config(text=f"Prediction: {pred}")

    avg_img = average_neighbors(img_flat, k)
    update_heatmap(0, avg_img)

    diff = composite_neighbors_diff(img_flat, img32, k)
    update_heatmap(1, diff)

    votes = voting_visualization(img_flat, k)
    update_vote_chart(votes)

# -----------------------------
# START
# -----------------------------
update_preview32(np.zeros((32, 32)))
update_heatmap(0, np.zeros((32, 32)))
update_heatmap(1, np.zeros((32, 32)))

root.mainloop()
