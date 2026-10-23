import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 屏蔽metric警告（zero_division=0带来）
warnings.filterwarnings("ignore")

# ===================== 全局参数 =====================
RANDOM_STATE = 42
N_SPLITS = 5
C = 10
gamma = "scale"
outDir = "ML_Models/SVM_RBF_5Fold"
os.makedirs(outDir, exist_ok=True)

# ============================================================
# 1. 读取数据
# ============================================================
data = sio.loadmat("../")
X = data["X"]
Y = data["Y"].ravel()

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Classes:", np.unique(Y))
classes = np.unique(Y)

# ============================================================
# 2. 5-Fold Stratified Cross-Validation
# ============================================================
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# 保存每一折结果
fold_acc = []
fold_precision = []
fold_recall = []
fold_f1 = []
fold_specificity = []

all_precision = []
all_recall = []
all_f1 = []
all_specificity = []

all_Y_test = []
all_Y_pred = []

# ============================================================
# 3. 5折循环
# ============================================================
for fold, (train_idx, test_idx) in enumerate(kf.split(X, Y), start=1):
    print("\n")
    print("=" * 70)
    print(f"                         Fold {fold}")
    print("=" * 70)

    X_train_raw, Y_train = X[train_idx], Y[train_idx]
    X_test_raw, Y_test = X[test_idx], Y[test_idx]

    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples : {len(test_idx)}")

    # -------------------------- SVM必须标准化！--------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 保存当前fold的scaler与测试集
    joblib.dump(scaler, os.path.join(outDir, f"scaler_fold_{fold}.pkl"))
    test_set = {"X_test_raw": X_test_raw, "X_test_scaled": X_test, "Y_test": Y_test, "test_idx": test_idx}
    joblib.dump(test_set, os.path.join(outDir, f"test_{fold}.pkl"))

    # SVM-RBF模型
    svm_model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train, Y_train)

    # -------------------------- 训练集指标 --------------------------
    train_pred = svm_model.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)
    train_precision = precision_score(Y_train, train_pred, average="macro", zero_division=0)
    train_recall = recall_score(Y_train, train_pred, average="macro", zero_division=0)
    train_f1 = f1_score(Y_train, train_pred, average="macro", zero_division=0)

    # -------------------------- 测试集预测 --------------------------
    test_pred = svm_model.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    test_precision = precision_score(Y_test, test_pred, average="macro", zero_division=0)
    test_recall = recall_score(Y_test, test_pred, average="macro", zero_division=0)
    test_f1 = f1_score(Y_test, test_pred, average="macro", zero_division=0)

    cm = confusion_matrix(Y_test, test_pred, labels=classes)
    print("\nTest Confusion Matrix:")
    print(cm)

    # -------------------------- 逐类别指标计算 --------------------------
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    specificity_per_class = []

    for i, _ in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        precision_class = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        recall_class = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1_class = 2*precision_class*recall_class/(precision_class+recall_class) if (precision_class+recall_class)>0 else 0.0
        specificity_class = tn/(tn+fp) if (tn+fp) > 0 else 0.0

        precision_per_class.append(precision_class)
        recall_per_class.append(recall_class)
        f1_per_class.append(f1_class)
        specificity_per_class.append(specificity_class)

    precision_per_class = np.array(precision_per_class)
    recall_per_class = np.array(recall_per_class)
    f1_per_class = np.array(f1_per_class)
    specificity_per_class = np.array(specificity_per_class)
    test_specificity = np.mean(specificity_per_class)

    # 保存fold指标
    fold_acc.append(test_acc)
    fold_precision.append(test_precision)
    fold_recall.append(test_recall)
    fold_f1.append(test_f1)
    fold_specificity.append(test_specificity)

    all_precision.append(precision_per_class)
    all_recall.append(recall_per_class)
    all_f1.append(f1_per_class)
    all_specificity.append(specificity_per_class)

    all_Y_test.extend(Y_test)
    all_Y_pred.extend(test_pred)

    # -------------------------- 打印当前fold结果 --------------------------
    print("\nTraining Set:")
    print(f"Accuracy     : {train_acc:.4f}")
    print(f"Precision    : {train_precision:.4f}")
    print(f"Recall       : {train_recall:.4f}")
    print(f"F1-score     : {train_f1:.4f}")

    print("\nTest Set:")
    print(f"Accuracy     : {test_acc:.4f}")
    print(f"Precision    : {test_precision:.4f}")
    print(f"Recall       : {test_recall:.4f}")
    print(f"F1-score     : {test_f1:.4f}")
    print(f"Specificity  : {test_specificity:.4f}")

    print("\nPer-Class Test Metrics:")
    print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Specificity':<15}")
    for i, class_label in enumerate(classes):
        print(f"{str(class_label):<10}{precision_per_class[i]:<15.4f}{recall_per_class[i]:<15.4f}{f1_per_class[i]:<15.4f}{specificity_per_class[i]:<15.4f}")

    # 保存fold预测结果 & 模型
    prediction_result = {
        "Y_test": Y_test, "Y_pred": test_pred, "test_idx": test_idx,
        "test_acc": test_acc, "test_precision": test_precision,
        "test_recall": test_recall, "test_f1": test_f1,
        "test_specificity": test_specificity,
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
        "per_class_f1": f1_per_class,
        "per_class_specificity": specificity_per_class,
        "confusion_matrix": cm, "classes": classes
    }
    joblib.dump(prediction_result, os.path.join(outDir, f"prediction_fold_{fold}.pkl"))
    joblib.dump(svm_model, os.path.join(outDir, f"SVM_RBF_model_fold_{fold}.pkl"))


# ============================================================
# 4. 汇总5折结果
# ============================================================
fold_acc = np.array(fold_acc)
fold_precision = np.array(fold_precision)
fold_recall = np.array(fold_recall)
fold_f1 = np.array(fold_f1)
fold_specificity = np.array(fold_specificity)

all_precision = np.array(all_precision)
all_recall = np.array(all_recall)
all_f1 = np.array(all_f1)
all_specificity = np.array(all_specificity)

all_Y_test = np.array(all_Y_test)
all_Y_pred = np.array(all_Y_pred)

# Mean ± SD (样本标准差 ddof=1)
mean_acc, std_acc = np.mean(fold_acc), np.std(fold_acc, ddof=1)
mean_precision, std_precision = np.mean(fold_precision), np.std(fold_precision, ddof=1)
mean_recall, std_recall = np.mean(fold_recall), np.std(fold_recall, ddof=1)
mean_f1, std_f1 = np.mean(fold_f1), np.std(fold_f1, ddof=1)
mean_specificity, std_specificity = np.mean(fold_specificity), np.std(fold_specificity, ddof=1)

# 打印5折汇总
print("\n")
print("=" * 75)
print("                 5-FOLD CV RESULTS")
print("=" * 75)
print(f"\nMacro Accuracy:    {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Macro Precision:   {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Macro Recall:      {mean_recall:.4f} ± {std_recall:.4f}")
print(f"Macro F1-score:    {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Macro Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")

# 打印每折单独结果
print("\n")
print("=" * 75)
print("                 EACH FOLD RESULTS")
print("=" * 75)
for i in range(N_SPLITS):
    print(f"\nFold {i+1}:")
    print(f"Accuracy     : {fold_acc[i]:.4f}")
    print(f"Precision    : {fold_precision[i]:.4f}")
    print(f"Recall       : {fold_recall[i]:.4f}")
    print(f"F1-score     : {fold_f1[i]:.4f}")
    print(f"Specificity  : {fold_specificity[i]:.4f}")

# 逐类别 Mean ± SD
print("\n")
print("=" * 90)
print("             PER-CLASS TEST METRICS: MEAN ± SD")
print("=" * 90)
print(f"{'Class':<10}{'Precision':<22}{'Recall':<22}{'F1-score':<22}{'Specificity':<22}")
perclass_summary = []
for i, class_label in enumerate(classes):
    p_m = np.mean(all_precision[:,i])
    p_s = np.std(all_precision[:,i], ddof=1)
    r_m = np.mean(all_recall[:,i])
    r_s = np.std(all_recall[:,i], ddof=1)
    f_m = np.mean(all_f1[:,i])
    f_s = np.std(all_f1[:,i], ddof=1)
    s_m = np.mean(all_specificity[:,i])
    s_s = np.std(all_specificity[:,i], ddof=1)
    print(f"{str(class_label):<10}{p_m:.4f} ± {p_s:.4f}   {r_m:.4f} ± {r_s:.4f}   {f_m:.4f} ± {f_s:.4f}   {s_m:.4f} ± {s_s:.4f}")
    perclass_summary.append({"Class":class_label,
                             "Precision_mean":p_m,"Precision_std":p_s,
                             "Recall_mean":r_m,"Recall_std":r_s,
                             "F1_mean":f_m,"F1_std":f_s,
                             "Specificity_mean":s_m,"Specificity_std":s_s})

# ============================================================
# 5. Pooled Confusion Matrix
# ============================================================
pooled_cm = confusion_matrix(all_Y_test, all_Y_pred, labels=classes)
row_sums = pooled_cm.sum(axis=1, keepdims=True)
pooled_cm_percent = np.divide(pooled_cm.astype(float), row_sums, out=np.zeros_like(pooled_cm, dtype=float), where=row_sums!=0)

print("\n")
print("=" * 70)
print("              POOLED CONFUSION MATRIX")
print("=" * 70)
print(pooled_cm)
print("\n")
print("=" * 70)
print("          POOLED CONFUSION MATRIX (%)")
print("=" * 70)
print(pooled_cm_percent)

# Pooled整体逐类别指标
overall_precision = []
overall_recall = []
overall_f1 = []
overall_specificity = []
for i, _ in enumerate(classes):
    tp = pooled_cm[i,i]
    fp = np.sum(pooled_cm[:,i]) - tp
    fn = np.sum(pooled_cm[i,:]) - tp
    tn = np.sum(pooled_cm) - tp - fp - fn

    precision_class = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall_class = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1_class = 2*precision_class*recall_class/(precision_class+recall_class) if (precision_class+recall_class)>0 else 0.0
    specificity_class = tn/(tn+fp) if (tn+fp)>0 else 0.0

    overall_precision.append(precision_class)
    overall_recall.append(recall_class)
    overall_f1.append(f1_class)
    overall_specificity.append(specificity_class)

overall_precision = np.array(overall_precision)
overall_recall = np.array(overall_recall)
overall_f1 = np.array(overall_f1)
overall_specificity = np.array(overall_specificity)

print("\n")
print("=" * 90)
print("                OVERALL PER-CLASS METRICS (Pooled)")
print("=" * 90)
print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Specificity':<15}")
pooled_perclass = []
for i, class_label in enumerate(classes):
    print(f"{str(class_label):<10}{overall_precision[i]:<15.4f}{overall_recall[i]:<15.4f}{overall_f1[i]:<15.4f}{overall_specificity[i]:<15.4f}")
    pooled_perclass.append({"Class":class_label,
                            "Precision":overall_precision[i],
                            "Recall":overall_recall[i],
                            "F1":overall_f1[i],
                            "Specificity":overall_specificity[i]})

# ============================================================
# 6. 绘制Pooled混淆矩阵图
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=pooled_cm_percent, display_labels=classes)
disp.plot(cmap="Blues", values_format=".1%", ax=ax, colorbar=False)

ax.set_title("SVM-RBF 5-Fold Pooled Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
ax.set_ylabel("True Label", fontsize=14, fontweight="bold")

for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.tick_params(axis="both", labelsize=12)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
for text in disp.text_.ravel():
    text.set_fontsize(14)
    text.set_fontweight("bold")

plt.tight_layout()
plt.savefig(os.path.join(outDir, "SVM_RBF_5Fold_pooled_confusion_matrix_percent.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. 保存全部结果
# ============================================================
cv_results = {
    "fold_accuracy": fold_acc,
    "fold_precision": fold_precision,
    "fold_recall": fold_recall,
    "fold_f1": fold_f1,
    "fold_specificity": fold_specificity,

    "mean_accuracy": mean_acc, "std_accuracy": std_acc,
    "mean_precision": mean_precision, "std_precision": std_precision,
    "mean_recall": mean_recall, "std_recall": std_recall,
    "mean_f1": mean_f1, "std_f1": std_f1,
    "mean_specificity": mean_specificity, "std_specificity": std_specificity,

    "all_precision": all_precision,
    "all_recall": all_recall,
    "all_f1": all_f1,
    "all_specificity": all_specificity,

    "pooled_confusion_matrix": pooled_cm,
    "pooled_confusion_matrix_percent": pooled_cm_percent,

    "overall_precision": overall_precision,
    "overall_recall": overall_recall,
    "overall_f1": overall_f1,
    "overall_specificity": overall_specificity,

    "classes": classes,
    "C": C, "gamma": gamma, "kernel": "rbf"
}
joblib.dump(cv_results, os.path.join(outDir, "SVM_RBF_5Fold_results.pkl"))

all_predictions = {"Y_test": all_Y_test, "Y_pred": all_Y_pred}
joblib.dump(all_predictions, os.path.join(outDir, "all_test_predictions.pkl"))

# ============ 导出CSV（论文作图直接用） ============
# 1. Fold level macro metrics
df_fold = pd.DataFrame({
    "Fold":np.arange(1,6),
    "Accuracy":fold_acc,
    "Precision":fold_precision,
    "Recall":fold_recall,
    "F1":fold_f1,
    "Specificity":fold_specificity
})
df_fold.to_csv(os.path.join(outDir, "fold_metrics.csv"), index=False)

# 2. Per class mean±sd
df_perclass = pd.DataFrame(perclass_summary)
df_perclass.to_csv(os.path.join(outDir, "perclass_mean_std.csv"), index=False)

# 3. Pooled per class metrics
df_pooled_perclass = pd.DataFrame(pooled_perclass)
df_pooled_perclass.to_csv(os.path.join(outDir, "pooled_perclass_metrics.csv"), index=False)

# 4. Pooled confusion matrix
np.savetxt(os.path.join(outDir,"pooled_cm.csv"), pooled_cm, delimiter=",")
np.savetxt(os.path.join(outDir,"pooled_cm_percent.csv"), pooled_cm_percent, delimiter=",")


print("\n")
print("=" * 75)
print("All 5-Fold results saved successfully. CSV exported for plotting.")
print("=" * 75)
print(f"\nOutput directory: {outDir}")