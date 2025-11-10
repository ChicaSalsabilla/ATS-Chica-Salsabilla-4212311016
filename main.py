"""
Program: Klasifikasi Huruf Tulisan Tangan (EMNIST Letters)
Menggunakan: HOG Feature Extraction + SVM
Evaluasi: Leave-One-Out Cross Validation (LOOCV)
"""

# ============================================================
# 1. Import Library
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.model_selection import LeaveOneOut
from skimage.feature import hog
import struct
import os

# ============================================================
# 2. Fungsi untuk membaca file EMNIST
# ============================================================
def load_emnist_letters(label_file, image_file, limit=None):
    """
    Membaca dataset EMNIST Letters dari file .ubyte
    """
    with open(label_file, 'rb') as lbpath:
        magic, num = struct.unpack(">II", lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(image_file, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), rows, cols)

    if limit:
        images = images[:limit]
        labels = labels[:limit]

    return images, labels

# ============================================================
# 3. Load dataset EMNIST Letters
# ============================================================
DATA_DIR = "Dataset"
LABEL_FILE = os.path.join(DATA_DIR, "emnist-letters-train-labels-idx1-ubyte")
IMAGE_FILE = os.path.join(DATA_DIR, "emnist-letters-train-images-idx3-ubyte")

print("📂 Memuat dataset EMNIST Letters...")
images, labels = load_emnist_letters(LABEL_FILE, IMAGE_FILE)
print(f"Total data tersedia: {len(labels)} sampel")

# ============================================================
# 4. Sampling data 13.000 (26 kelas × 500)
# ============================================================
n_classes = 26
samples_per_class = 500
selected_images = []
selected_labels = []

print("🔹 Melakukan sampling data seimbang (26 kelas × 500)...")

for i in range(1, n_classes + 1):
    idx = np.where(labels == i)[0]
    chosen = np.random.choice(idx, samples_per_class, replace=False)
    selected_images.append(images[chosen])
    selected_labels.append(labels[chosen])

X = np.vstack(selected_images)
y = np.hstack(selected_labels)

print(f"Dataset akhir: {X.shape}, Label: {y.shape}")

# ============================================================
# 5. Ekstraksi fitur HOG
# ============================================================
print("🔹 Mengekstrak fitur HOG...")
hog_features = []
for img in tqdm(X, desc="Ekstraksi HOG"):
    fd = hog(img, orientations=9, pixels_per_cell=(8,8),
             cells_per_block=(2,2), block_norm='L2-Hys')
    hog_features.append(fd)

hog_features = np.array(hog_features)
print("Ekstraksi selesai. Bentuk fitur:", hog_features.shape)

# ============================================================
# 6. Leave-One-Out Cross Validation (LOOCV)
# ============================================================
print("🔹 Evaluasi menggunakan Leave-One-Out Cross Validation...")

loo = LeaveOneOut()
clf = svm.SVC(kernel='rbf', C=10, gamma=0.01)

y_true_all = []
y_pred_all = []

# ⚠️ Gunakan subset agar cepat diuji
# Setelah berhasil jalan, ubah N = 13000 untuk evaluasi penuh
N = 13000
print(f"Menjalankan LOOCV pada {N} sampel (demo)...")

for train_idx, test_idx in tqdm(loo.split(hog_features[:N], y[:N]), total=N):
    X_train, X_test = hog_features[train_idx], hog_features[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    y_true_all.append(y_test[0])
    y_pred_all.append(y_pred[0])

# ============================================================
# 7. Evaluasi performa
# ============================================================
print("\n📊 Evaluasi hasil klasifikasi...")
acc = accuracy_score(y_true_all, y_pred_all)
prec = precision_score(y_true_all, y_pred_all, average='macro')
f1 = f1_score(y_true_all, y_pred_all, average='macro')
cm = confusion_matrix(y_true_all, y_pred_all)

print(f"✅ Accuracy  : {acc:.4f}")
print(f"✅ Precision : {prec:.4f}")
print(f"✅ F1-Score  : {f1:.4f}")

# ============================================================
# 8. Visualisasi Confusion Matrix
# ============================================================
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", cbar=False)
plt.title("Confusion Matrix - EMNIST Letters (HOG + SVM, LOOCV)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("\nProgram selesai ✅")
