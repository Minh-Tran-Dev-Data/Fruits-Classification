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
    features = [float(mean_h.flatten()[0]), float(std_h.flatten()[0]),
                float(mean_s.flatten()[0]), float(std_s.flatten()[0]),
                float(mean_v.flatten()[0]), float(std_v.flatten()[0])]
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


## ---------------------------------------------------------------------
## GHEP 3 KY THUAT + PHAN LOAI
## ---------------------------------------------------------------------
import os
from collections import Counter


def auto_mask(img, s_thresh=30, v_thresh=200):
    hsv, h, s, v = hsv_procession(img)
    mask_nen = cv2.inRange(hsv, (0, 0, v_thresh), (179, s_thresh, 255))
    mask = cv2.bitwise_not(mask_nen)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_all_features(img, mask=None):
    if mask is None:
        mask = auto_mask(img)

    feat_hsv = extract_hsv(img, mask=mask)
    feat_hist = extract_histogram(img, mask=mask)
    feat_stats = extract_stats(img, mask=mask)
    return feat_hsv, feat_hist, feat_stats


def build_reference_db(image_paths_by_label):
    db_raw = {}
    for label, paths in image_paths_by_label.items():
        entries = []
        for p in paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                print(f"[Canh bao] Khong doc duoc anh: {p}")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            f_hsv, f_hist, f_stats = extract_all_features(img)
            entries.append({"path": p, "hsv": f_hsv, "hist": f_hist, "stats": f_stats})
        db_raw[label] = entries
    return db_raw


def compute_normalization_stats(db_raw):
    all_hsv, all_hist, all_stats = [], [], []
    for entries in db_raw.values():
        for e in entries:
            all_hsv.append(e["hsv"])
            all_hist.append(e["hist"])
            all_stats.append(e["stats"])

    all_hsv = np.array(all_hsv)
    all_hist = np.array(all_hist)
    all_stats = np.array(all_stats)
    hist_std = np.maximum(all_hist.std(axis=0), 0.01)

    return {
        "hsv_mean": all_hsv.mean(axis=0), "hsv_std": all_hsv.std(axis=0) + 1e-6,
        "hist_mean": all_hist.mean(axis=0), "hist_std": hist_std,
        "stats_mean": all_stats.mean(axis=0), "stats_std": all_stats.std(axis=0) + 1e-6,
        # luu lai so chieu tung phan de sau nay tach vector ra giai thich duoc
        "n_hsv": all_hsv.shape[1], "n_hist": all_hist.shape[1], "n_stats": all_stats.shape[1],
    }


def _zscore(x, mean, std, clip=5.0):
    z = (x - mean) / std
    return np.clip(z, -clip, clip)


def build_normalized_db(db_raw, norm_params):
    db_norm = {}
    for label, entries in db_raw.items():
        items = []
        for e in entries:
            v_hsv = _zscore(e["hsv"], norm_params["hsv_mean"], norm_params["hsv_std"])
            v_hist = _zscore(e["hist"], norm_params["hist_mean"], norm_params["hist_std"])
            v_stats = _zscore(e["stats"], norm_params["stats_mean"], norm_params["stats_std"])
            vector = np.concatenate([v_hsv, v_hist, v_stats])
            items.append({"path": e["path"], "vector": vector})
        db_norm[label] = items
    return db_norm


def _feat_test_vector(img, norm_params, mask=None):
    f_hsv, f_hist, f_stats = extract_all_features(img, mask=mask)
    v_hsv = _zscore(f_hsv, norm_params["hsv_mean"], norm_params["hsv_std"])
    v_hist = _zscore(f_hist, norm_params["hist_mean"], norm_params["hist_std"])
    v_stats = _zscore(f_stats, norm_params["stats_mean"], norm_params["stats_std"])
    return np.concatenate([v_hsv, v_hist, v_stats])


def classify_fruit(img, db_norm, norm_params, mask=None, k=3):
    if mask is None:
        mask = auto_mask(img)
    feat_test = _feat_test_vector(img, norm_params, mask=mask)

    all_dists = []
    for label, items in db_norm.items():
        for item in items:
            dist = np.linalg.norm(feat_test - item["vector"])
            all_dists.append((label, dist))

    all_dists.sort(key=lambda x: x[1])
    top_k = all_dists[:k]

    votes = Counter(label for label, _ in top_k)
    best_label = votes.most_common(1)[0][0]
    confidence = votes[best_label] / k

    label_avg_dist = {
        label: float(np.mean([d for lb, d in all_dists if lb == label]))
        for label in db_norm
    }
    return best_label, confidence, label_avg_dist


def classify_fruit_detailed(img, db_norm, norm_params, mask=None, k=3):
    if mask is None:
        mask = auto_mask(img)
    feat_test = _feat_test_vector(img, norm_params, mask=mask)

    n_hsv = norm_params["n_hsv"]
    n_hist = norm_params["n_hist"]

    all_matches = []
    for label, items in db_norm.items():
        for item in items:
            v = item["vector"]
            dist = float(np.linalg.norm(feat_test - v))
            all_matches.append({"label": label, "path": item["path"], "dist": dist, "vector": v})

    all_matches.sort(key=lambda x: x["dist"])
    top_k = all_matches[:k]

    votes = Counter(m["label"] for m in top_k)
    best_label = votes.most_common(1)[0][0]
    confidence = votes[best_label] / k

    label_avg_dist = {
        label: float(np.mean([m["dist"] for m in all_matches if m["label"] == label]))
        for label in db_norm
    }

    nearest = all_matches[0]
    v_near = nearest["vector"]
    d_hsv = float(np.linalg.norm(feat_test[:n_hsv] - v_near[:n_hsv]))
    d_hist = float(np.linalg.norm(feat_test[n_hsv:n_hsv + n_hist] - v_near[n_hsv:n_hsv + n_hist]))
    d_stats = float(np.linalg.norm(feat_test[n_hsv + n_hist:] - v_near[n_hsv + n_hist:]))

    detail = {
        "nearest_path": nearest["path"],
        "nearest_label": nearest["label"],
        "nearest_total_dist": nearest["dist"],
        "breakdown": {"hsv": d_hsv, "hist": d_hist, "stats": d_stats},
        "top_k": [{"label": m["label"], "path": m["path"], "dist": m["dist"]} for m in top_k],
    }
    return best_label, confidence, label_avg_dist, detail


def build_pipeline_from_folder(root_folder, max_images_per_class=None):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths_by_label = {}
    for label in sorted(os.listdir(root_folder)):
        label_dir = os.path.join(root_folder, label)
        if not os.path.isdir(label_dir):
            continue
        files = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))
                  if f.lower().endswith(valid_ext)]
        if max_images_per_class:
            files = files[:max_images_per_class]
        image_paths_by_label[label] = files

    db_raw = build_reference_db(image_paths_by_label)
    norm_params = compute_normalization_stats(db_raw)
    db_norm = build_normalized_db(db_raw, norm_params)
    return db_norm, norm_params
