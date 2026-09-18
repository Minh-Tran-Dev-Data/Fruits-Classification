import cv2
import numpy as np
import matplotlib.pyplot as plt

from hsv_module import hsv_procession


def histogram_procession(img, mask=None, bins=32):
    hsv, h, s, v = hsv_procession(img)

    hist_h = cv2.calcHist([h], [0], mask, [bins], [0, 180])
    hist_s = cv2.calcHist([s], [0], mask, [bins], [0, 256])
    hist_v = cv2.calcHist([v], [0], mask, [bins], [0, 256])

    hist_h = cv2.normalize(hist_h, None).flatten()
    hist_s = cv2.normalize(hist_s, None).flatten()
    hist_v = cv2.normalize(hist_v, None).flatten()

    return hist_h, hist_s, hist_v


def extract_histogram(img, mask=None, bins=32):
    hist_h, hist_s, hist_v = histogram_procession(img, mask=mask, bins=bins)
    features = np.concatenate([hist_h, hist_s, hist_v])
    return features.astype(np.float32)


def compare_histogram(img1, img2, mask1=None, mask2=None, bins=32):
    hist_h1, hist_s1, hist_v1 = histogram_procession(img1, mask=mask1, bins=bins)
    hist_h2, hist_s2, hist_v2 = histogram_procession(img2, mask=mask2, bins=bins)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Anh 1')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Anh 2')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(hist_h1, color='darkorange', label='H')
    plt.plot(hist_s1, color='seagreen', label='S')
    plt.plot(hist_v1, color='steelblue', label='V')
    plt.title('Histogram anh 1')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(hist_h2, color='darkorange', label='H')
    plt.plot(hist_s2, color='seagreen', label='S')
    plt.plot(hist_v2, color='steelblue', label='V')
    plt.title('Histogram anh 2')
    plt.legend()

    plt.tight_layout()
    plt.show()

    correl_h = cv2.compareHist(hist_h1, hist_h2, cv2.HISTCMP_CORREL)
    correl_s = cv2.compareHist(hist_s1, hist_s2, cv2.HISTCMP_CORREL)
    correl_v = cv2.compareHist(hist_v1, hist_v2, cv2.HISTCMP_CORREL)

    print(f"Correlation kenh H: {correl_h:.4f}")
    print(f"Correlation kenh S: {correl_s:.4f}")
    print(f"Correlation kenh V: {correl_v:.4f}")

    return correl_h, correl_s, correl_v
