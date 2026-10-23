import scipy.io as sio
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, log_loss)

# ============================================================
# 0. 参数设置
# ============================================================
DATA_PATH = "../"
RANDOM_STATE = 35
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_BOOTSTRAP = 5000
BOOTSTRAP_RANDOM_STATE = 2026
N_SAMPLES_PER_CLASS = 32
GROUP_SIZE = 2
OUTPUT_DIR = "./Nested_Stratifed_Group_CV_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 70)
print("Loading dataset...")
print("=" * 70)

data = sio.loadmat(DATA_PATH)
X = data["X"]
Y = data["Y"].ravel()

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Classes:", np.unique(Y))

classes = np.sort(np.unique(Y))
n_classes = len(classes)

print("\nClass distribution:")
for c in classes:
    print(f"Class {c}: {np.sum(Y == c)} samples")

# ============================================================
# 2. 构建 Group
# ============================================================
print("\n" + "=" * 70)
print("Constructing groups...")
print("=" * 70)

groups = np.zeros(len(Y), dtype=int)
group_id = 0

for c in classes:
    class_idx = np.where(Y == c)[0]
    if len(class_idx) != N_SAMPLES_PER_CLASS:
        raise ValueError(f"Class {c} has {len(class_idx)} samples, but expected {N_SAMPLES_PER_CLASS}.")
    for i in range(0, N_SAMPLES_PER_CLASS, GROUP_SIZE):
        pair_idx = class_idx[i:i + GROUP_SIZE]
        if len(pair_idx) != GROUP_SIZE:
            raise ValueError(f"Class {c}: incomplete group at position {i}.")
        groups[pair_idx] = group_id
        print(f"Class {c}: samples {i + 1}-{i + GROUP_SIZE} -> Group {group_id}")
        group_id += 1

n_groups = len(np.unique(groups))
print("\nTotal groups:", n_groups)
if n_groups != 64:
    raise ValueError(f"Expected 64 groups, but got {n_groups}.")

# ============================================================
# 3. 检查每个group
# ============================================================
print("\n" + "=" * 70)
print("Checking group structure...")
print("=" * 70)

unique_groups, group_counts = np.unique(groups, return_counts=True)
print("Group sizes:")
print(np.unique(group_counts, return_counts=True))
if not np.all(group_counts == 2):
    raise ValueError("Some groups do not contain exactly 2 samples.")

print("✓ Every group contains exactly 2 samples.")
print("✓ Every group belongs to only one class.")

# ============================================================
# 4. Nested Stratified Group CV
# ============================================================
outer_cv = StratifiedGroupKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedGroupKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
'''
# ============================================================
# 5. GBDT参数网格
# ============================================================
param_grid = {
    "n_estimators": [50, 100, 120],
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [3,5,7],
    "min_samples_leaf": [5,10,15]
}
'''
param_grid = {
    "n_estimators": [120],
    "learning_rate": [0.02],
    "max_depth": [2],
    "min_samples_leaf": [15]
}
# ============================================================
# 6. 初始化结果存储
# ============================================================
fold_results = []
oof_true = np.zeros(len(Y), dtype=Y.dtype)
oof_pred = np.zeros(len(Y), dtype=Y.dtype)
oof_proba = np.zeros((len(Y), n_classes), dtype=float)
oof_fold = np.zeros(len(Y), dtype=int)

# ============================================================
# 7. Nested Group CV主循环
# ============================================================
print("\n" + "=" * 70)
print("Starting Nested Stratified Group CV")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groups), start=1):
    print("\n")
    print("#" * 70)
    print(f"OUTER FOLD {fold}/{N_OUTER_SPLITS}")
    print("#" * 70)

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    # group泄露检查
    train_groups_unique = np.unique(groups_train)
    test_groups_unique = np.unique(groups_test)
    overlap = np.intersect1d(train_groups_unique, test_groups_unique)
    print("\nGroup leakage check:")
    if len(overlap) > 0:
        raise ValueError(f"Group leakage detected! Overlapping groups: {overlap}")
    print("✓ No group overlap between train and test.")

    # 当前fold类别分布
    print("\nTraining samples per class:")
    for c in classes:
        print(f"Class {c}: {np.sum(Y_train == c)} samples")

    print("\nTest samples per class:")
    for c in classes:
        print(f"Class {c}: {np.sum(Y_test == c)} samples")

    # 内部网格搜索
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid,
                               scoring="f1_macro", cv=inner_cv, n_jobs=-1,
                               refit=True, return_train_score=True)
    grid_search.fit(X_train, Y_train, groups=groups_train)
    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)
    print("Best inner‑CV macro F1:", grid_search.best_score_)

    train_loss = best_model.train_score_
    test_loss = np.array([log_loss(Y_test, proba) for proba in best_model.staged_predict_proba(X_test)])

    Y_pred = best_model.predict(X_test)
    Y_proba = best_model.predict_proba(X_test)

    oof_true[test_idx] = Y_test
    oof_pred[test_idx] = Y_pred
    oof_proba[test_idx] = Y_proba
    oof_fold[test_idx] = fold

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    rec = recall_score(Y_test, Y_pred, average="macro", zero_division=0)
    f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(Y_test, Y_pred, labels=classes)

    print("\nFold metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # 保存模型
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"GBDT_fold{fold}.joblib"))

    fold_results.append({
        "fold": fold,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_groups": train_groups_unique,
        "test_groups": test_groups_unique,
        "best_params": grid_search.best_params_,
        "inner_cv_best_f1": grid_search.best_score_,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "train_loss": train_loss,
        "test_loss": test_loss
    })

# ============================================================
# 8. 5‑fold Mean ± SD
# ============================================================
print("\n")
print("=" * 70)
print("5‑FOLD CROSS‑VALIDATION RESULTS")
print("=" * 70)

acc_arr = np.array([r["accuracy"] for r in fold_results])
prec_arr = np.array([r["precision_macro"] for r in fold_results])
rec_arr = np.array([r["recall_macro"] for r in fold_results])
f1_arr = np.array([r["f1_macro"] for r in fold_results])

acc_mean, acc_sd = np.mean(acc_arr), np.std(acc_arr, ddof=1)
prec_mean, prec_sd = np.mean(prec_arr), np.std(prec_arr, ddof=1)
rec_mean, rec_sd = np.mean(rec_arr), np.std(rec_arr, ddof=1)
f1_mean, f1_sd = np.mean(f1_arr), np.std(f1_arr, ddof=1)

print(f"Accuracy : {acc_mean:.4f} ± {acc_sd:.4f}")
print(f"Precision: {prec_mean:.4f} ± {prec_sd:.4f}")
print(f"Recall   : {rec_mean:.4f} ± {rec_sd:.4f}")
print(f"F1       : {f1_mean:.4f} ± {f1_sd:.4f}")

# ============================================================
# 9. 每个类别5‑fold Mean ± SD
# ============================================================
print("\n" + "=" * 70)
print("PER‑CLASS 5‑FOLD RESULTS")
print("=" * 70)

prec_per_fold = np.array([precision_score(Y[r["test_idx"]], oof_pred[r["test_idx"]], labels=classes, average=None, zero_division=0) for r in fold_results])
rec_per_fold = np.array([recall_score(Y[r["test_idx"]], oof_pred[r["test_idx"]], labels=classes, average=None, zero_division=0) for r in fold_results])
f1_per_fold = np.array([f1_score(Y[r["test_idx"]], oof_pred[r["test_idx"]], labels=classes, average=None, zero_division=0) for r in fold_results])

for i, c in enumerate(classes):
    p_m = np.mean(prec_per_fold[:,i])
    p_s = np.std(prec_per_fold[:,i], ddof=1)
    r_m = np.mean(rec_per_fold[:,i])
    r_s = np.std(rec_per_fold[:,i], ddof=1)
    f_m = np.mean(f1_per_fold[:,i])
    f_s = np.std(f1_per_fold[:,i], ddof=1)
    print(f"\nClass {c}:")
    print(f"  Precision: {p_m:.4f} ± {p_s:.4f}")
    print(f"  Recall   : {r_m:.4f} ± {r_s:.4f}")
    print(f"  F1       : {f_m:.4f} ± {f_s:.4f}")

# ============================================================
# 10. Pooled OOF指标
# ============================================================
print("\n" + "=" * 70)
print("POOLED OOF RESULTS")
print("=" * 70)

oof_acc = accuracy_score(oof_true, oof_pred)
oof_prec = precision_score(oof_true, oof_pred, labels=classes, average="macro", zero_division=0)
oof_rec = recall_score(oof_true, oof_pred, labels=classes, average="macro", zero_division=0)
oof_f1 = f1_score(oof_true, oof_pred, labels=classes, average="macro", zero_division=0)
oof_cm = confusion_matrix(oof_true, oof_pred, labels=classes)

print(f"OOF Accuracy : {oof_acc:.4f}")
print(f"OOF Precision: {oof_prec:.4f}")
print(f"OOF Recall   : {oof_rec:.4f}")
print(f"OOF F1       : {oof_f1:.4f}")
print("\nPooled OOF Confusion Matrix:")
print(oof_cm)

oof_prec_cls = precision_score(oof_true, oof_pred, labels=classes, average=None, zero_division=0)
oof_rec_cls = recall_score(oof_true, oof_pred, labels=classes, average=None, zero_division=0)
oof_f1_cls = f1_score(oof_true, oof_pred, labels=classes, average=None, zero_division=0)
'''
# ============================================================
# 11. Group‑level Bootstrap 95% CI
# ============================================================
print("\n" + "=" * 70)
print("GROUP‑LEVEL BOOTSTRAP 95% CI")
print("=" * 70)

def group_bootstrap_accuracy_ci(y_true, y_pred, groups, n_bootstrap=5000, random_state=2026):
    rng = np.random.default_rng(random_state)
    ug = np.unique(groups)
    n_g = len(ug)
    boot_acc = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sel_g = rng.choice(ug, size=n_g, replace=True)
        mask = np.isin(groups, sel_g)
        boot_acc[b] = accuracy_score(y_true[mask], y_pred[mask])
    lower = np.percentile(boot_acc, 2.5)
    upper = np.percentile(boot_acc, 97.5)
    return boot_acc, lower, upper

boot_acc_arr, ci_low, ci_high = group_bootstrap_accuracy_ci(oof_true, oof_pred, groups, N_BOOTSTRAP, BOOTSTRAP_RANDOM_STATE)

print(f"OOF Accuracy = {oof_acc:.4f}")
print(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
print(f"95% CI (%) = [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
'''
# ============================================================
# 11. Wilson 95% CI for pooled OOF accuracy
#     Based on 128 OOF test samples
# ============================================================
print("\n" + "=" * 70)
print("WILSON 95% CI FOR POOLED OOF ACCURACY")
print("=" * 70)

def wilson_accuracy_ci(y_true, y_pred, confidence=0.95):
    """
    Wilson score confidence interval for classification accuracy.
    Treat each OOF test sample as a Bernoulli trial:
        success = correctly classified
        failure = incorrectly classified
    """
    n = len(y_true)
    correct = np.sum(y_true == y_pred)
    # Observed accuracy
    p_hat = correct / n
    # z value for 95% CI
    z = 1.959963984540054
    # Wilson score interval
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (
        z * np.sqrt(
            p_hat * (1 - p_hat) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )
    lower = center - margin
    upper = center + margin
    return p_hat, correct, n, lower, upper

# Use all 128 out-of-fold predictions
oof_acc_wilson, n_correct, n_total, ci_low, ci_high = \
    wilson_accuracy_ci(oof_true, oof_pred)

print(f"Total OOF test samples : {n_total}")
print(f"Correct predictions    : {n_correct}")
print(f"Incorrect predictions  : {n_total - n_correct}")
print(f"OOF Accuracy           : {oof_acc_wilson:.4f}")
print(f"95% Wilson CI          : [{ci_low:.4f}, {ci_high:.4f}]")
print(f"95% Wilson CI (%)      : [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
'''
# ============================================================
# 11. Group-level Bootstrap 95% CI
#     Accuracy / Precision / Recall / F1
# ============================================================
print("\n" + "=" * 70)
print("GROUP-LEVEL BOOTSTRAP 95% CI")
print("=" * 70)

def group_bootstrap_metrics_ci(
        y_true, y_pred, groups, classes, n_bootstrap=5000, random_state=2026):
    rng = np.random.default_rng(random_state)
    # 所有group
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    # Bootstrap结果
    boot_acc = np.zeros(n_bootstrap)
    boot_prec = np.zeros(n_bootstrap)
    boot_rec = np.zeros(n_bootstrap)
    boot_f1 = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # 1. 以group为单位进行有放回抽样
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        # 2. 获取被抽中的group对应的所有样本
        bootstrap_indices = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
       # bootstrap_indices = np.isin(groups, sampled_groups)
        y_true_boot = y_true[bootstrap_indices]
        y_pred_boot = y_pred[bootstrap_indices]
        # 3. 计算四个指标
        boot_acc[b] = accuracy_score(y_true_boot, y_pred_boot)
        boot_prec[b] = precision_score(y_true_boot, y_pred_boot, labels=classes, average="macro", zero_division=0)
        boot_rec[b] = recall_score(y_true_boot, y_pred_boot, labels=classes, average="macro", zero_division=0)
        boot_f1[b] = f1_score(y_true_boot, y_pred_boot, labels=classes, average="macro", zero_division=0)

    # 4. 计算95% CI
    acc_low, acc_high = np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5)
    prec_low, prec_high = np.percentile(boot_prec, 2.5), np.percentile(boot_prec, 97.5)
    rec_low, rec_high = np.percentile(boot_rec, 2.5), np.percentile(boot_rec, 97.5)
    f1_low, f1_high = np.percentile(boot_f1, 2.5), np.percentile(boot_f1, 97.5)

    return (boot_acc, acc_low, acc_high, boot_prec, prec_low, prec_high,
            boot_rec, rec_low, rec_high, boot_f1, f1_low, f1_high)

# ============================================================
# Run group-level bootstrap
# ============================================================
(boot_acc_arr, acc_ci_low, acc_ci_high,
 boot_prec_arr, prec_ci_low, prec_ci_high,
 boot_rec_arr, rec_ci_low, rec_ci_high,
 boot_f1_arr, f1_ci_low, f1_ci_high) = group_bootstrap_metrics_ci(
    oof_true, oof_pred, groups, classes, N_BOOTSTRAP, BOOTSTRAP_RANDOM_STATE)

# ============================================================
# Print results
# ============================================================
print(f"\nNumber of groups = {len(np.unique(groups))}")
print(f"Number of OOF samples = {len(oof_true)}")
print(f"Number of bootstrap repetitions = {N_BOOTSTRAP}")

print("\nPooled OOF Results:")
print(f"Accuracy : {oof_acc:.4f}")
print(f"Precision: {oof_prec:.4f}")
print(f"Recall   : {oof_rec:.4f}")
print(f"F1       : {oof_f1:.4f}")

print("\nGroup-level Bootstrap 95% CI:")
print(f"Accuracy : [{acc_ci_low:.4f}, {acc_ci_high:.4f}] ({acc_ci_low*100:.2f}%, {acc_ci_high*100:.2f}%)")
print(f"Precision: [{prec_ci_low:.4f}, {prec_ci_high:.4f}] ({prec_ci_low*100:.2f}%, {prec_ci_high*100:.2f}%)")
print(f"Recall   : [{rec_ci_low:.4f}, {rec_ci_high:.4f}] ({rec_ci_low*100:.2f}%, {rec_ci_high*100:.2f}%)")
print(f"F1       : [{f1_ci_low:.4f}, {f1_ci_high:.4f}] ({f1_ci_low*100:.2f}%, {f1_ci_high*100:.2f}%)")
'''
# ============================================================
# 12. 绘制混淆矩阵
# ============================================================
plt.figure(figsize=(7,6))
disp = ConfusionMatrixDisplay(confusion_matrix=oof_cm, display_labels=classes)
disp.plot(cmap="Blues", values_format="d")
plt.title("Pooled Out‑of‑Fold Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "OOF_Confusion_Matrix.png"), dpi=600, bbox_inches="tight")
plt.close()

oof_cm_norm = oof_cm.astype(float)/oof_cm.sum(axis=1, keepdims=True)
print("\nRow‑normalized OOF confusion matrix (%):")
print(np.round(oof_cm_norm*100,2))

plt.figure(figsize=(7,6))
disp2 = ConfusionMatrixDisplay(confusion_matrix=oof_cm_norm, display_labels=classes)
disp2.plot(cmap="Blues", values_format=".2f")
plt.title("Pooled OOF Confusion Matrix (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "OOF_Confusion_Matrix_Normalized.png"), dpi=600, bbox_inches="tight")
plt.close()

# ============================================================
# 13. 各fold loss曲线
# ============================================================
for res in fold_results:
    plt.figure(figsize=(7,5))
    x = np.arange(1, len(res["train_loss"])+1)
    plt.plot(x, res["train_loss"], label="Training Loss")
    plt.plot(x, res["test_loss"], label="Test Loss")
    plt.xlabel("Number of Boosting Iterations")
    plt.ylabel("Log Loss")
    plt.title(f"Fold {res['fold']}: Training/Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fold{res['fold']}_Loss.png"), dpi=600, bbox_inches="tight")
    plt.close()

# ============================================================
# 14. 保存全部结果
# ============================================================
final_results = {
    "X_shape": X.shape,
    "Y": Y,
    "classes": classes,
    "groups": groups,
    "n_groups": n_groups,
    "n_outer_splits": N_OUTER_SPLITS,
    "n_inner_splits": N_INNER_SPLITS,
    "random_state": RANDOM_STATE,
    "fold_results": fold_results,
    "accuracy_mean": acc_mean, "accuracy_sd": acc_sd,
    "precision_mean": prec_mean, "precision_sd": prec_sd,
    "recall_mean": rec_mean, "recall_sd": rec_sd,
    "f1_mean": f1_mean, "f1_sd": f1_sd,
    "oof_true": oof_true, "oof_pred": oof_pred, "oof_proba": oof_proba, "oof_fold": oof_fold,
    "oof_accuracy": oof_acc, "oof_precision": oof_prec, "oof_recall": oof_rec, "oof_f1": oof_f1,
    "oof_confusion_matrix": oof_cm, "oof_confusion_matrix_normalized": oof_cm_norm,
    "oof_precision_per_class": oof_prec_cls,
    "oof_recall_per_class": oof_rec_cls,
    "oof_f1_per_class": oof_f1_cls,
    '''
    "n_bootstrap": N_BOOTSTRAP,
    "bootstrap_accuracy": boot_acc_arr,
    "accuracy_ci_95_lower": ci_low,
    "accuracy_ci_95_upper": ci_high,
 
    "n_bootstrap": N_BOOTSTRAP,

    "bootstrap_accuracy": boot_acc_arr,
    "bootstrap_precision": boot_prec_arr,
    "bootstrap_recall": boot_rec_arr,
    "bootstrap_f1": boot_f1_arr,

    "accuracy_ci_95_lower": acc_ci_low,
    "accuracy_ci_95_upper": acc_ci_high,

    "precision_ci_95_lower": prec_ci_low,
    "precision_ci_95_upper": prec_ci_high,

    "recall_ci_95_lower": rec_ci_low,
    "recall_ci_95_upper": rec_ci_high,

    "f1_ci_95_lower": f1_ci_low,
    "f1_ci_95_upper": f1_ci_high,
    
    '''
    "n_oof_samples": n_total,
    "n_correct": n_correct,
    "accuracy_ci_95_lower": ci_low,
    "accuracy_ci_95_upper": ci_high,
    "accuracy_ci_method": "Wilson score interval"


}

res_path = os.path.join(OUTPUT_DIR, "nested_stratified_group_cv_results.pkl")
joblib.dump(final_results, res_path)

# ============================================================
# 15. 最终汇总输出
# ============================================================
print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"5‑fold Accuracy: {acc_mean*100:.2f}% ± {acc_sd*100:.2f}%")
print(f"5‑fold Precision: {prec_mean*100:.2f}% ± {prec_sd*100:.2f}%")
print(f"5‑fold Recall: {rec_mean*100:.2f}% ± {rec_sd*100:.2f}%")
print(f"5‑fold F1: {f1_mean*100:.2f}% ± {f1_sd*100:.2f}%")
'''
print(f"\nPooled OOF Accuracy: {oof_acc*100:.2f}%")
print(f"Group‑bootstrap 95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
'''
print(f"\nPooled OOF Accuracy: {oof_acc*100:.2f}%")
print(f"Wilson 95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
'''
print("\nPooled OOF Results with 95% CI:")
print(f"Accuracy : {oof_acc*100:.2f}% "
      f"[{acc_ci_low*100:.2f}%, {acc_ci_high*100:.2f}%]")

print(f"Precision: {oof_prec*100:.2f}% "
      f"[{prec_ci_low*100:.2f}%, {prec_ci_high*100:.2f}%]")

print(f"Recall   : {oof_rec*100:.2f}% "
      f"[{rec_ci_low*100:.2f}%, {rec_ci_high*100:.2f}%]")

print(f"F1       : {oof_f1*100:.2f}% "
      f"[{f1_ci_low*100:.2f}%, {f1_ci_high*100:.2f}%]")
'''
print("\nResults saved to:")
print(res_path)
print("\nDone.")