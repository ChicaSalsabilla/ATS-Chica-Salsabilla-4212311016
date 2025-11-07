import numpy as np
import cv2
from scipy.signal import correlate2d, convolve2d
import os

# Gunakan fungsi ini untuk mapping nilai pixel ke 0-255
def map_to_8bit(img_in):
    img_out = None
    img_out = ((img_in - img_in.min()) * 255/(img_in.max() - img_in.min())).astype(np.uint8)
    return img_out

def correlate(img_in, kernel):
    # Lengkapi fungsi berikut ini agar menghasilkan luaran berupa hasil correlation filter dengan parameter sebagai berikut
    # stride = 1
    # full padding
    # img_in : merupakan image input dalam bentuk BGR image
    # kernel : merupakan filter yang digunakan, ukuran dapat bervariasi (3x3, 5x5, 7x7, etc)
    # pada fungsi di bawah ini, lakukan konversi img_in dari BGR ke Grayscale terlebih dahulu, kemudian kalkulasi hasil correlation

    # TODO: Implementasikan fungsi untuk melakukan operasi correlate
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    kh, kw = kernel.shape
    pad_h = kh - 1
    pad_w = kw - 1

    # Full padding
    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1

    output = np.zeros((out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        for x in range(out_w):
            region = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(region * kernel)

    return output

def convolve(img_in, kernel):
    # Lengkapi fungsi berikut ini agar menghasilkan luaran berupa hasil convolution dengan parameter sebagai berikut
    # stride = 1
    # same padding
    # kernel : merupakan filter yang digunakan, ukuran dapat bervariasi (3x3, 5x5, 7x7, etc)
    # pada fungsi di bawah ini, lakukan konversi img_in dari BGR ke Grayscale terlebih dahulu, kemudian kalkulasi hasil correlation

    # TODO: Implementasikan fungsi untuk melakukan operasi convolve
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    out_h, out_w = gray.shape
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Flip kernel untuk convolution
    flipped_kernel = np.flipud(np.fliplr(kernel))

    for y in range(out_h):
        for x in range(out_w):
            region = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(region * flipped_kernel)

    return output

def main():
    image_path = "C:/Users/Asus/assigment/p3-correlation-and-convolution-ChicaSalsabilla/tests/cat-image.jpg"
    bgr_image = cv2.imread(image_path)
    save_dir = "C:/Users/Asus/assigment/p3-correlation-and-convolution-ChicaSalsabilla/Hasil-filter"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Blur filter (mean filter)
    blur_kernel = np.ones((5, 5), dtype=np.float32) / 25.0

    # Sharpen filter
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)

    # Sobel filter (edge detection)
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    #Manual
    manual_blur = convolve(bgr_image, blur_kernel)
    manual_sharpen = convolve(bgr_image, sharpen_kernel)
    manual_edge_x = convolve(bgr_image, sobel_x)
    manual_edge_y = convolve(bgr_image, sobel_y)
    manual_edge = np.sqrt(manual_edge_x**2 + manual_edge_y**2)

    #OpenCV 
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cv_blur = cv2.filter2D(gray, cv2.CV_32F, blur_kernel)
    cv_sharpen = cv2.filter2D(gray, cv2.CV_32F, sharpen_kernel)
    cv_edge_x = cv2.filter2D(gray, cv2.CV_32F, sobel_x)
    cv_edge_y = cv2.filter2D(gray, cv2.CV_32F, sobel_y)
    cv_edge = np.sqrt(cv_edge_x**2 + cv_edge_y**2)

    cv_blur = map_to_8bit(cv_blur)
    cv_sharpen = map_to_8bit(cv_sharpen)
    cv_edge = map_to_8bit(cv_edge)

    #Hasil
    cv2.imshow("Original", gray)
    cv2.imshow("Manual Blur", map_to_8bit(manual_blur))
    cv2.imshow("OpenCV Blur", cv_blur)
    cv2.imshow("Manual Sharpen", map_to_8bit(manual_sharpen))
    cv2.imshow("OpenCV Sharpen", cv_sharpen)
    cv2.imshow("Manual Edge Detection", map_to_8bit(manual_edge))
    cv2.imshow("OpenCV Edge Detection", cv_edge)

    #Simpan hasil
    cv2.imwrite(os.path.join(save_dir, "manual_blur.png"), map_to_8bit(manual_blur))
    cv2.imwrite(os.path.join(save_dir, "opencv_blur.png"), cv_blur)
    cv2.imwrite(os.path.join(save_dir, "manual_sharpen.png"), map_to_8bit(manual_sharpen))
    cv2.imwrite(os.path.join(save_dir, "opencv_sharpen.png"), cv_sharpen)
    cv2.imwrite(os.path.join(save_dir, "manual_edge.png"), map_to_8bit(manual_edge))
    cv2.imwrite(os.path.join(save_dir, "opencv_edge.png"), cv_edge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Perbandingan Nilai
    diff_blur = np.abs(manual_blur - cv_blur).mean()
    diff_sharpen = np.abs(manual_sharpen - cv_sharpen).mean()
    diff_edge = np.abs(manual_edge - cv_edge).mean()

    print("🔍 Perbandingan hasil manual vs OpenCV:")
    print(f"- Blur difference     : {diff_blur:.4f}")
    print(f"- Sharpen difference  : {diff_sharpen:.4f}")
    print(f"- Edge difference     : {diff_edge:.4f}")

if __name__ == "__main__":
    main()