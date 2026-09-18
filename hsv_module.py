import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Tach kenh H, S, V tu anh RGB
# ---------------------------------------------------------------------------
def hsv_procession(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    return img_hsv, h, s, v


# ---------------------------------------------------------------------------
# 2. Phan doan theo nguong HSV -> tao mask + anh da tach nen
# ---------------------------------------------------------------------------
def hsv_segment(img, h_low, h_high, s_low, s_high, v_low, v_high):
    hsv, h, s, v = hsv_procession(img)

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)

    # lam sach nhieu (tuy chon nhung nen co)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segmented = cv2.bitwise_and(img, img, mask=mask)
    return mask, segmented


# ---------------------------------------------------------------------------
# 3. Trich dac trung HSV (mean/std) - CHI tren vung mask (neu co)
# ---------------------------------------------------------------------------
def extract_hsv(img, mask=None):
    hsv, h, s, v = hsv_procession(img)

    mean_h, std_h = cv2.meanStdDev(h, mask=mask)
    mean_s, std_s = cv2.meanStdDev(s, mask=mask)
    mean_v, std_v = cv2.meanStdDev(v, mask=mask)

    features = [float(mean_h.flatten()[0]), float(std_h.flatten()[0]),
                float(mean_s.flatten()[0]), float(std_s.flatten()[0]),
                float(mean_v.flatten()[0]), float(std_v.flatten()[0])]
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. So sanh truc quan 2 anh: hien anh goc, mask, anh da tach nen, histogram H
# ---------------------------------------------------------------------------
def compare_hsv(img1, img2, mask1=None, mask2=None):
    hsv1, h1, s1, v1 = hsv_procession(img1)
    hsv2, h2, s2, v2 = hsv_procession(img2)

    hist1 = cv2.calcHist([h1], [0], mask1, [180], [0, 180])
    hist1 = cv2.normalize(hist1, None).flatten()
    hist2 = cv2.calcHist([h2], [0], mask2, [180], [0, 180])
    hist2 = cv2.normalize(hist2, None).flatten()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Anh 1 (RGB)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Anh 2 (RGB)')
    plt.axis('off')

    # Hien thi rieng kenh H (mau xam) thay vi imshow ca anh HSV
    # vi imshow luon hieu du lieu la RGB, hien HSV truc tiep se bi sai mau
    plt.subplot(2, 2, 3)
    plt.plot(hist1, color='darkorange', label='Anh 1')
    plt.plot(hist2, color='royalblue', label='Anh 2')
    plt.title('Histogram kenh H')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.imshow(h1, cmap='gray')
    plt.title('Kenh H - Anh 1')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    correl = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f"Do tuong quan histogram kenh H: {correl:.4f} (cang gan 1 cang giong)")
    return correl


# ---------------------------------------------------------------------------
# Vi du chay thu (tach rieng khoi phan dinh nghia ham)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    img1 = cv2.cvtColor(cv2.imread('data/apple1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('data/apple2.jpg'), cv2.COLOR_BGR2RGB)

    mask1, seg1 = hsv_segment(img1, 0, 10, 70, 255, 50, 255)
    mask2, seg2 = hsv_segment(img2, 0, 10, 70, 255, 50, 255)

    feat1 = extract_hsv(img1, mask=mask1)
    feat2 = extract_hsv(img2, mask=mask2)
    print("Dac trung HSV anh 1:", feat1)
    print("Dac trung HSV anh 2:", feat2)

    compare_hsv(img1, img2, mask1=mask1, mask2=mask2)
