import os
import cv2
import numpy as np

from hsv_module import hsv_segment, extract_hsv
from histogram_module import extract_histogram
from stats_module import extract_stats


# ---------------------------------------------------------------------------
# 1. Tao mask tu dong (khong can biet truoc mau qua) - dung cho nen trang
#    (vd dataset fruits-360). Neu anh nen phuc tap, thay bang hsv_segment
#    voi nguong H/S/V nguoi dung tu chinh qua GUI.
# ---------------------------------------------------------------------------
def auto_mask(img, s_thresh=30, v_thresh=200):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_nen = cv2.inRange(hsv, (0, 0, v_thresh), (179, s_thresh, 255))
    mask = cv2.bitwise_not(mask_nen)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------------------------
# 2. Trich toan bo dac trung (HSV + Histogram + Thong ke) cho 1 anh
# ---------------------------------------------------------------------------
def extract_all_features(img, mask=None):
    if mask is None:
        mask = auto_mask(img)

    feat_hsv = extract_hsv(img, mask=mask)
    feat_hist = extract_histogram(img, mask=mask)
    feat_stats = extract_stats(img, mask=mask)

    # tra ve rieng tung phan de con chuan hoa/can trong so doc lap sau nay
    return feat_hsv, feat_hist, feat_stats


# ---------------------------------------------------------------------------
# 3. Xay "ngan hang" dac trung tham chieu tu cac anh mau da biet nhan
# ---------------------------------------------------------------------------
def build_reference_db(image_paths_by_label):
    """
    image_paths_by_label: dict {'tao': [path1, path2, ...], 'chuoi': [...], ...}
    Tra ve: db_raw (dict label -> list cac tuple (hsv, hist, stats) tung anh mau)
    """
    db_raw = {}
    for label, paths in image_paths_by_label.items():
        feats = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[Canh bao] Khong doc duoc anh: {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feats.append(extract_all_features(img))
        db_raw[label] = feats
    return db_raw


# ---------------------------------------------------------------------------
# 4. Tinh tham so chuan hoa (mean/std toan cuc) tren TAT CA anh mau
#    -> de dam bao HSV/Histogram/Thong ke duoc "can bang" khi so khoang cach
# ---------------------------------------------------------------------------
def compute_normalization_stats(db_raw):
    all_hsv, all_hist, all_stats = [], [], []
    for feats in db_raw.values():
        for f_hsv, f_hist, f_stats in feats:
            all_hsv.append(f_hsv)
            all_hist.append(f_hist)
            all_stats.append(f_stats)

    all_hsv = np.array(all_hsv)
    all_hist = np.array(all_hist)
    all_stats = np.array(all_stats)

    # QUAN TRONG: dung "san" (floor) cho std thay vi +1e-6 don thuan.
    # Ly do: nhieu bin histogram co the LUON BANG 0 o TAT CA anh mau (vd:
    # khong anh mau nao co mau xanh la), khien std ~ 0. Neu anh test lai
    # co gia tri khac 0 dung bin do (vd gap qua mau la thuc su), z-score
    # se bi CHIA CHO SO GAN 0 -> ra so khong lo, lam sai lech toan bo
    # khoang cach (mot dac trung "hong" se at het cac dac trung con lai).
    # hist da duoc normalize ve khoang 0-1 nen dung san 0.01 la hop ly.
    hist_std = np.maximum(all_hist.std(axis=0), 0.01)

    norm_params = {
        "hsv_mean": all_hsv.mean(axis=0), "hsv_std": all_hsv.std(axis=0) + 1e-6,
        "hist_mean": all_hist.mean(axis=0), "hist_std": hist_std,
        "stats_mean": all_stats.mean(axis=0), "stats_std": all_stats.std(axis=0) + 1e-6,
    }
    return norm_params


def _zscore(x, mean, std, clip=5.0):
    z = (x - mean) / std
    # Gioi han (clip) z-score trong [-clip, clip]: tranh 1 vai dac trung
    # bat thuong (outlier) chi phoi toan bo khoang cach tong hop.
    return np.clip(z, -clip, clip)


# ---------------------------------------------------------------------------
# 5. Xay db da chuan hoa (dung de so sanh) tu db_raw + norm_params
# ---------------------------------------------------------------------------
def build_normalized_db(db_raw, norm_params):
    db_norm = {}
    for label, feats in db_raw.items():
        vecs = []
        for f_hsv, f_hist, f_stats in feats:
            v_hsv = _zscore(f_hsv, norm_params["hsv_mean"], norm_params["hsv_std"])
            v_hist = _zscore(f_hist, norm_params["hist_mean"], norm_params["hist_std"])
            v_stats = _zscore(f_stats, norm_params["stats_mean"], norm_params["stats_std"])
            vecs.append(np.concatenate([v_hsv, v_hist, v_stats]))
        db_norm[label] = np.array(vecs)
    return db_norm


# ---------------------------------------------------------------------------
# 6. HAM PHAN LOAI CHINH - so anh moi voi tung anh mau, chon nhan gan nhat (KNN)
# ---------------------------------------------------------------------------
def classify_fruit(img, db_norm, norm_params, mask=None, k=3):
    """
    img: anh RGB can phan loai
    db_norm: ket qua tu build_normalized_db
    norm_params: ket qua tu compute_normalization_stats
    k: so lang gieng gan nhat dung de "vote" (KNN)

    Tra ve: nhan du doan, do tin cay (0-1), bang khoang cach trung binh toi tung loai
    """
    if mask is None:
        mask = auto_mask(img)

    f_hsv, f_hist, f_stats = extract_all_features(img, mask=mask)
    v_hsv = _zscore(f_hsv, norm_params["hsv_mean"], norm_params["hsv_std"])
    v_hist = _zscore(f_hist, norm_params["hist_mean"], norm_params["hist_std"])
    v_stats = _zscore(f_stats, norm_params["stats_mean"], norm_params["stats_std"])
    feat_test = np.concatenate([v_hsv, v_hist, v_stats])

    # tinh khoang cach toi TAT CA anh mau cua TAT CA loai
    all_dists = []   # list (label, distance)
    for label, vecs in db_norm.items():
        for v in vecs:
            dist = np.linalg.norm(feat_test - v)
            all_dists.append((label, dist))

    all_dists.sort(key=lambda x: x[1])         # sap xep gan nhat truoc
    top_k = all_dists[:k]                      # lay k lang gieng gan nhat

    # "vote": nhan nao xuat hien nhieu nhat trong top_k thi thang
    from collections import Counter
    votes = Counter(label for label, _ in top_k)
    best_label = votes.most_common(1)[0][0]

    # do tin cay = ty le vote cua nhan thang trong top_k
    confidence = votes[best_label] / k

    # bang khoang cach trung binh toi tung loai (de tham khao/debug)
    label_avg_dist = {}
    for label in db_norm:
        dists_label = [d for lb, d in all_dists if lb == label]
        label_avg_dist[label] = float(np.mean(dists_label))

    return best_label, confidence, label_avg_dist


# ---------------------------------------------------------------------------
# 7. Ham tien ich: xay dung toan bo pipeline tu 1 thu muc anh mau co cau truc
#    thu_muc/ten_loai/anh.jpg
# ---------------------------------------------------------------------------
def build_pipeline_from_folder(root_folder, max_images_per_class=None):
    """
    root_folder co cau truc:
        root_folder/
            Apple/
                img1.jpg
                img2.jpg
            Banana/
                img1.jpg
                ...
    Tra ve: db_norm, norm_params (dung truc tiep cho classify_fruit)
    """
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


# ---------------------------------------------------------------------------
# Vi du chay thu
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    image_paths_by_label = {
        "tao": ["data/01_apple.jpg", "data/06_apple.jpg", "data/11_apple.jpg"],
        "chuoi": ["data/02_banana.jpg", "data/07_banana.jpg", "data/12_banana.jpg"],
        "cam": ["data/03_orange.jpg", "data/08_orange.jpg", "data/13_orange.jpg"],
    }

    db_raw = build_reference_db(image_paths_by_label)
    norm_params = compute_normalization_stats(db_raw)
    db_norm = build_normalized_db(db_raw, norm_params)

    img_test = cv2.cvtColor(cv2.imread("data/04_lime.jpg"), cv2.COLOR_BGR2RGB)
    label, conf, dist_table = classify_fruit(img_test, db_norm, norm_params, k=3)

    print(f"Du doan: {label} (do tin cay: {conf:.2f})")
    print("Khoang cach trung binh toi tung loai:")
    for lb, d in sorted(dist_table.items(), key=lambda x: x[1]):
        print(f"  {lb:10s}: {d:.3f}")
