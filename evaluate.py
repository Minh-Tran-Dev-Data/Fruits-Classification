import os
import cv2
import numpy as np
import csv
from classify_fruit import (
    build_reference_db,
    compute_normalization_stats,
    build_normalized_db,
    classify_fruit,
)
# ---------------------------------------------------------------------------
# 1. Lay N anh tu 1 thu muc, bat dau tu vi tri "skip"
#    (dung de tach TRAIN va TEST khong bi trung anh)
# ---------------------------------------------------------------------------
def get_n_images(folder, n=20, skip=0):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(valid_ext)]
    files = files[skip:skip + n]
    return [os.path.join(folder, f) for f in files]


# ---------------------------------------------------------------------------
# 2. Danh gia: chay classify_fruit tren TOAN BO anh test, so voi nhan thuc te
# ---------------------------------------------------------------------------
def evaluate_accuracy(test_paths_by_label, db_norm, norm_params, k=3, verbose=True):
    """
    test_paths_by_label: dict {'tao': [path1, path2, ...], 'chuoi': [...], ...}
        CAC ANH NAY PHAI KHAC voi anh dung de build_reference_db (train),
        neu khong accuracy se bi "ao" (cao gia tao vi tu so voi chinh minh).

    Tra ve:
        accuracy tong the (0-1)
        ket qua chi tiet tung anh (list dict)
        confusion matrix (dict long-form: {(nhan_thuc, nhan_du_doan): so_luong})
    """
    results = []
    labels = list(test_paths_by_label.keys())
    confusion = {(a, b): 0 for a in labels for b in labels}

    total, correct = 0, 0
    for true_label, paths in test_paths_by_label.items():
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[Canh bao] Khong doc duoc anh: {p}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_label, conf, _ = classify_fruit(img, db_norm, norm_params, k=k)

            # neu classify_fruit tra ve nhan la ("tao", 0.8) ma khong co trong
            # danh sach labels test (vi du du doan sai thanh loai khong test) ->
            # van tinh la SAI, nhung khong lam crash confusion matrix
            if (true_label, pred_label) not in confusion:
                confusion[(true_label, pred_label)] = 0
            confusion[(true_label, pred_label)] += 1

            is_correct = (pred_label == true_label)
            correct += int(is_correct)
            total += 1

            results.append({
                "file": p,
                "nhan_thuc_te": true_label,
                "nhan_du_doan": pred_label,
                "do_tin_cay": round(conf, 2),
                "dung_sai": "DUNG" if is_correct else "SAI",
            })

            if verbose:
                status = "DUNG" if is_correct else "SAI "
                print(f"[{status}] {os.path.basename(p):20s} "
                      f"thuc_te={true_label:10s} du_doan={pred_label:10s} "
                      f"tin_cay={conf:.2f}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results, confusion


# ---------------------------------------------------------------------------
# 3. In confusion matrix dang bang de de doc
# ---------------------------------------------------------------------------
def print_confusion_matrix(confusion, labels):
    col_w = max(10, max(len(l) for l in labels) + 2)
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print("\nConfusion Matrix (hang = thuc te, cot = du doan):")
    print(header)
    for true_label in labels:
        row = f"{true_label:<{col_w}}"
        for pred_label in labels:
            count = confusion.get((true_label, pred_label), 0)
            row += f"{count:>{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# 4. Accuracy rieng cho tung loai (giup thay loai nao yeu, can cai thien)
# ---------------------------------------------------------------------------
def per_class_accuracy(confusion, labels):
    print("\nAccuracy theo tung loai:")
    for label in labels:
        total_label = sum(confusion.get((label, p), 0) for p in labels)
        correct_label = confusion.get((label, label), 0)
        acc = correct_label / total_label if total_label > 0 else 0.0
        print(f"  {label:<10s}: {correct_label}/{total_label} = {acc*100:.1f}%")


# ---------------------------------------------------------------------------
# 5. (Tuy chon) Ve confusion matrix bang matplotlib - de dua vao bao cao
# ---------------------------------------------------------------------------
def plot_confusion_matrix(confusion, labels, save_path=None):
    import matplotlib.pyplot as plt

    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = confusion.get((true_label, pred_label), 0)

    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 2))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Nhan du doan")
    ax.set_ylabel("Nhan thuc te")
    ax.set_title("Confusion Matrix")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Da luu confusion matrix: {save_path}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Vi du chay toan bo (sua duong dan cho khop du lieu cua ban)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = "data"
    fruit_classes = ["Apple", "Banana", "Orange"]

    image_paths_by_label = {'dataset/apple/anh-qua-tao.jpg','dataset/apple/tao_xanh.jpg'}
    test_paths_by_label = {'dataset/apple/tao_xanh.jpg'}

    for cls in fruit_classes:
        folder = os.path.join(base, cls)
        image_paths_by_label[cls] = get_n_images(folder, n=20, skip=0)
        test_paths_by_label[cls] = get_n_images(folder, n=15, skip=20)

    print("Dang xay ngan hang dac trung (train)...")
    db_raw = build_reference_db(image_paths_by_label)
    norm_params = compute_normalization_stats(db_raw)
    db_norm = build_normalized_db(db_raw, norm_params)

    print("\nDang danh gia tren tap test...\n")
    accuracy, results, confusion = evaluate_accuracy(test_paths_by_label, db_norm, norm_params, k=3)

    print(f"\n=== TONG KET ===")
    print(f"Accuracy tong the: {accuracy*100:.2f}%  ({sum(r['dung_sai']=='DUNG' for r in results)}/{len(results)})")

    labels = list(test_paths_by_label.keys())
    print_confusion_matrix(confusion, labels)
    per_class_accuracy(confusion, labels)

    # Xuat ket qua chi tiet ra CSV de dua vao bao cao

    with open("output/evaluation_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nDa luu bao cao chi tiet: output/evaluation_report.csv")

    # Ve va luu confusion matrix (bo qua neu khong can hien thi hinh)
    try:
        plot_confusion_matrix(confusion, labels, save_path="output/confusion_matrix.png")
    except Exception as e:
        print(f"Khong ve duoc confusion matrix: {e}")
