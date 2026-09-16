import cv2
import numpy as np
import matplotlib.pyplot as plt

## HSV
def hsv_procession(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    return img_hsv, h, s, v


def hsv_segment(img, h_low, h_high, s_low, s_high, v_low, v_high):
    hsv, h, s, v = hsv_procession(img)
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    segmented = cv2.bitwise_and(img, img, mask=mask)
    return mask, segmented


def extract_hsv(img, mask=None):
    hsv, h, s, v = hsv_procession(img)
    mean_h, std_h = cv2.meanStdDev(h, mask=mask)
    mean_s, std_s = cv2.meanStdDev(s, mask=mask)
    mean_v, std_v = cv2.meanStdDev(v, mask=mask)
    features = [float(mean_h), float(std_h),
                float(mean_s), float(std_s),
                float(mean_v), float(std_v)]
    return np.array(features, dtype=np.float32)
def compare_hsv (img1,img2):
    hsv_rgb = cv2.cvtColor(img2,cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(img1)
    plt.title('ảnh RGB')
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(img2)
    plt.title('ảnh HSV ')
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(hsv_rgb)
    plt.title('HSV chuyển về RGB')
    plt.axis('off')
    plt.show()
## histogram
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
## Statictis
def stats_procession(img):
    r, g, b = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return r, g, b, gray

def extract_stats(img, mask=None):
    r, g, b, gray = stats_procession(img)

    mean_r, std_r = cv2.meanStdDev(r, mask=mask)
    mean_g, std_g = cv2.meanStdDev(g, mask=mask)
    mean_b, std_b = cv2.meanStdDev(b, mask=mask)
    mean_gray, std_gray = cv2.meanStdDev(gray, mask=mask)

    features = [float(mean_r.flatten()[0]), float(std_r.flatten()[0]),
                float(mean_g.flatten()[0]), float(std_g.flatten()[0]),
                float(mean_b.flatten()[0]), float(std_b.flatten()[0]),
                float(mean_gray.flatten()[0]), float(std_gray.flatten()[0])]
    return np.array(features, dtype=np.float32)

def compute_full_statistics(img, mask=None):
    total_pixels = img.shape[0] * img.shape[1]
    feat = extract_stats(img, mask=mask)

    if mask is not None and mask.any():
        area_pixels = int(np.count_nonzero(mask))
        num_objects, boxes = _count_objects(mask)
    else:
        area_pixels = total_pixels
        num_objects, boxes = 0, []

    stats = {
        "kich_thuoc_anh": f"{img.shape[1]}x{img.shape[0]}",
        "tong_so_pixel": total_pixels,
        "so_pixel_vung_chon": area_pixels,
        "ty_le_dien_tich_%": round(100.0 * area_pixels / total_pixels, 2),
        "mean_R": round(float(feat[0]), 2), "std_R": round(float(feat[1]), 2),
        "mean_G": round(float(feat[2]), 2), "std_G": round(float(feat[3]), 2),
        "mean_B": round(float(feat[4]), 2), "std_B": round(float(feat[5]), 2),
        "mean_Gray": round(float(feat[6]), 2), "std_Gray": round(float(feat[7]), 2),
        "so_doi_tuong_phat_hien": num_objects,
    }
    return stats, boxes


def _count_objects(mask, min_area=150):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    return len(boxes), boxes

def compare_stats(img1, img2, mask1=None, mask2=None):
    stats1 = extract_stats(img1, mask=mask1)
    stats2 = extract_stats(img2, mask=mask2)
    labels = ['Mean R', 'Std R', 'Mean G', 'Std G',
              'Mean B', 'Std B', 'Mean Gray', 'Std Gray']

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.title('Anh 1')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.title('Anh 2')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, stats1, width, label='Anh 1', color='indianred')
    plt.bar(x + width / 2, stats2, width, label='Anh 2', color='cadetblue')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.title('So sanh thong ke giua 2 anh')

    plt.tight_layout()
    plt.show()

    print(f"{'Chi so':<12}{'Anh 1':>10}{'Anh 2':>10}")
    for name, v1, v2 in zip(labels, stats1, stats2):
        print(f"{name:<12}{v1:>10.2f}{v2:>10.2f}")

    return stats1, stats2
