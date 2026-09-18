import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Tach kenh R, G, B, Gray tu anh RGB
# ---------------------------------------------------------------------------
def stats_procession(img):
    r, g, b = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return r, g, b, gray


# ---------------------------------------------------------------------------
# 2. Trich dac trung thong ke (mean/std) - CHI tren vung mask (neu co)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Thong ke chi tiet (dung cho hien thi tren GUI / bao cao) - co dien tich, so doi tuong
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. So sanh truc quan thong ke giua 2 anh
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Vi du chay thu
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from hsv_module import hsv_segment   # dung lai ham HSV da co de tao mask

    img1 = cv2.cvtColor(cv2.imread('data/apple1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('data/banana1.jpg'), cv2.COLOR_BGR2RGB)

    mask1, _ = hsv_segment(img1, 0, 10, 70, 255, 50, 255)
    mask2, _ = hsv_segment(img2, 20, 35, 70, 255, 70, 255)

    feat1 = extract_stats(img1, mask=mask1)
    feat2 = extract_stats(img2, mask=mask2)
    print("Dac trung thong ke anh 1:", feat1)
    print("Dac trung thong ke anh 2:", feat2)

    stats1, boxes1 = compute_full_statistics(img1, mask=mask1)
    print("Thong ke chi tiet anh 1:", stats1)

    compare_stats(img1, img2, mask1=mask1, mask2=mask2)
