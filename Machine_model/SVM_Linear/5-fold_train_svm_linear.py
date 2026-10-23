import scipy.io as sio
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 1. 读取数据
# ============================================================
data = sio.loadmat("../")
X = data["X"]
Y = data["Y"].ravel()
print("Dataset shape:")
print("X:", X.shape)
print("Y:", Y.shape)
print("Classes:", np.unique(Y))

# ============================================================
# 2. 输出文件夹
# ============================================================
outDir = "ML_Models/SVM_Linear_5Fold"
os.makedirs(outDir, exist_ok=True)

# ============================================================
# 3. Stratified 5-Fold Cross-Validation
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 4. SVM Linear 参数
# ============================================================
C = 150
random_state = 42

# ============================================================
# 5. 保存结果
# ============================================================
fold_metrics = []
all_test_true = []
all_test_pred = []
confusion_matrices = []
all_precision = []
all_recall = []
all_f1 = []
all_specificity = []
labels = np.unique(Y)

# ============================================================
# 6. 开始 5-Fold Cross-Validation
# ============================================================
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), start=1):
    print("\n")
    print("=" * 70)
    print(f"Fold {fold}/5")
    print("=" * 70)
    # ========================================================
    # 6.1 获取当前 Fold Train / Test
    # ========================================================
    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    # ========================================================
    # 6.2 建立 SVM Linear 模型
    # ========================================================
    svm_model = SVC(kernel="linear", C=C, probability=True, random_state=random_state)
    # ========================================================
    # 6.3 训练模型
    # ========================================================
    svm_model.fit(X_train, Y_train)
    print("\nSVM Linear training completed.")
    # ========================================================
    # 6.4 Training Set 预测
    # ========================================================
    train_pred = svm_model.predict(X_train)
    # ========================================================
    # 6.5 Test Set 预测
    # ========================================================
    test_pred = svm_model.predict(X_test)
    # ========================================================
    # 7. Training Set 指标
    # ========================================================
    train_acc = accuracy_score(Y_train, train_pred)
    train_precision = precision_score(Y_train, train_pred, average="macro", zero_division=0)
    train_recall = recall_score(Y_train, train_pred, average="macro", zero_division=0)
    train_f1 = f1_score(Y_train, train_pred, average="macro", zero_division=0)
    # ========================================================
    # 8. Test Set Confusion Matrix
    # ========================================================
    cm = confusion_matrix(Y_test, test_pred, labels=labels)
    confusion_matrices.append(cm)
    # ========================================================
    # 9. 计算每个类别的 Precision / Recall / F1 / Specificity
    # ========================================================
    n_classes = len(labels)
    prec_per_class = np.zeros(n_classes, dtype=float)
    rec_per_class = np.zeros(n_classes, dtype=float)
    f1_per_class = np.zeros(n_classes, dtype=float)
    spec_per_class = np.zeros(n_classes, dtype=float)
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        # Precision
        if (tp + fp) > 0:
            prec_per_class[i] = tp / (tp + fp)
        else:
            prec_per_class[i] = 0.0
        # Recall / Sensitivity
        if (tp + fn) > 0:
            rec_per_class[i] = tp / (tp + fn)
        else:
            rec_per_class[i] = 0.0
        # F1-score
        if (prec_per_class[i] + rec_per_class[i]) > 0:
            f1_per_class[i] = 2 * prec_per_class[i] * rec_per_class[i] / (prec_per_class[i] + rec_per_class[i])
        else:
            f1_per_class[i] = 0.0
        # Specificity
        if (tn + fp) > 0:
            spec_per_class[i] = tn / (tn + fp)
        else:
            spec_per_class[i] = 0.0
    # ========================================================
    # 10. Macro-Specificity
    # ========================================================
    test_specificity = np.mean(spec_per_class)
    # ========================================================
    # 11. 保存每个 Fold 每类别指标
    # ========================================================
    all_precision.append(prec_per_class)
    all_recall.append(rec_per_class)
    all_f1.append(f1_per_class)
    all_specificity.append(spec_per_class)
    # ========================================================
    # 12. Test Set Macro 指标
    # ========================================================
    test_acc = accuracy_score(Y_test, test_pred)
    test_precision = precision_score(Y_test, test_pred, average="macro", zero_division=0)
    test_recall = recall_score(Y_test, test_pred, average="macro", zero_division=0)
    test_f1 = f1_score(Y_test, test_pred, average="macro", zero_division=0)
    # ========================================================
    # 13. 输出 Training Set 结果
    # ========================================================
    print("\n----------------------------------------")
    print(f"Fold {fold} Training Set:")
    print("----------------------------------------")
    print(f"Accuracy    : {train_acc:.4f}")
    print(f"Precision   : {train_precision:.4f}")
    print(f"Recall      : {train_recall:.4f}")
    print(f"F1-score    : {train_f1:.4f}")
    # ========================================================
    # 14. 输出 Test Set Macro 结果
    # ========================================================
    print("\n----------------------------------------")
    print(f"Fold {fold} Test Set:")
    print("----------------------------------------")
    print(f"Accuracy    : {test_acc:.4f}")
    print(f"Precision   : {test_precision:.4f}")
    print(f"Recall      : {test_recall:.4f}")
    print(f"F1-score    : {test_f1:.4f}")
    print(f"Specificity : {test_specificity:.4f}")
    # ========================================================
    # 15. 输出每个类别的指标
    # ========================================================
    print("\nPer-class Test Metrics:")
    print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Specificity':<15}")
    print("-" * 70)
    for i, label in enumerate(labels):
        print(f"{str(label):<10}{prec_per_class[i]:<15.4f}{rec_per_class[i]:<15.4f}{f1_per_class[i]:<15.4f}{spec_per_class[i]:<15.4f}")
    # ========================================================
    # 16. 保存当前 Fold 指标
    # ========================================================
    fold_result = {"Fold": fold,
                   "Train_Accuracy": train_acc,
                   "Train_Precision_macro": train_precision,
                   "Train_Recall_macro": train_recall,
                   "Train_F1_macro": train_f1,
                   "Test_Accuracy": test_acc,
                   "Test_Precision_macro": test_precision,
                   "Test_Recall_macro": test_recall,
                   "Test_F1_macro": test_f1,
                   "Test_Specificity_macro": test_specificity,
                   "Test_Precision_per_class": prec_per_class,
                   "Test_Recall_per_class": rec_per_class,
                   "Test_F1_per_class": f1_per_class,
                   "Test_Specificity_per_class": spec_per_class,
                   "C": C}
    fold_metrics.append(fold_result)
    # ========================================================
    # 17. 输出原始混淆矩阵
    # ========================================================
    print("\nTest Confusion Matrix:")
    print(cm)
    # ========================================================
    # 18. 混淆矩阵转换为百分比，按真实类别行归一化
    # ========================================================
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    print("\nTest Confusion Matrix (%):")
    print(np.round(cm_percent * 100, 2))
    # ========================================================
    # 19. 保存当前 Fold 测试数据
    # ========================================================
    joblib.dump((X_test, Y_test), os.path.join(outDir, f"test_{fold}.pkl"))
    # ========================================================
    # 20. 保存当前 Fold 预测结果
    # ========================================================
    prediction_results = {"Y_test": Y_test, "Y_pred": test_pred, "test_indices": test_idx}
    joblib.dump(prediction_results, os.path.join(outDir, f"test_predictions_fold_{fold}.pkl"))
    # ========================================================
    # 21. 保存当前 Fold 指标
    # ========================================================
    fold_test_metrics = {"Accuracy": test_acc,
                         "Precision_macro": test_precision,
                         "Recall_macro": test_recall,
                         "F1_macro": test_f1,
                         "Specificity_macro": test_specificity,
                         "Precision_per_class": prec_per_class,
                         "Recall_per_class": rec_per_class,
                         "F1_per_class": f1_per_class,
                         "Specificity_per_class": spec_per_class,
                         "C": C}
    joblib.dump(fold_test_metrics, os.path.join(outDir, f"test_metrics_fold_{fold}.pkl"))
    # ========================================================
    # 22. 保存当前 Fold 原始混淆矩阵
    # ========================================================
    joblib.dump(cm, os.path.join(outDir, f"confusion_matrix_fold_{fold}.pkl"))
    # ========================================================
    # 23. 保存当前 Fold 百分比混淆矩阵
    # ========================================================
    joblib.dump(cm_percent, os.path.join(outDir, f"confusion_matrix_percent_fold_{fold}.pkl"))
    # ========================================================
    # 24. 绘制当前 Fold 百分比混淆矩阵
    # ========================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".1%", colorbar=False)
    ax.set_title(f"SVM Linear - Fold {fold} Test Confusion Matrix", fontname="Times New Roman", fontweight="bold", fontsize=14)
    ax.set_xlabel("Predicted Label", fontname="Times New Roman", fontweight="bold", fontsize=12)
    ax.set_ylabel("True Label", fontname="Times New Roman", fontweight="bold", fontsize=12)
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
        label.set_fontsize(11)
    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
        label.set_fontsize(11)
    for text in ax.texts:
        text.set_fontname("Times New Roman")
        text.set_fontweight("bold")
        text.set_fontsize(11)
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, f"SVM_Linear_test_confusion_matrix_percent_fold_{fold}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # ========================================================
    # 25. 保存当前 Fold 模型
    # ========================================================
    joblib.dump(svm_model, os.path.join(outDir, f"SVM_Linear_model_fold_{fold}.pkl"))
    # ========================================================
    # 26. 保存所有 Fold 测试预测
    # ========================================================
    all_test_true.extend(Y_test)
    all_test_pred.extend(test_pred)

# ============================================================
# 27. 转换为 numpy array
# ============================================================
all_test_true = np.array(all_test_true)
all_test_pred = np.array(all_test_pred)
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)
all_f1 = np.array(all_f1)
all_specificity = np.array(all_specificity)

# ============================================================
# 28. 提取 5 个 Fold 的 Test Macro 指标
# ============================================================
test_accuracy = np.array([x["Test_Accuracy"] for x in fold_metrics])
test_precision = np.array([x["Test_Precision_macro"] for x in fold_metrics])
test_recall = np.array([x["Test_Recall_macro"] for x in fold_metrics])
test_f1 = np.array([x["Test_F1_macro"] for x in fold_metrics])
test_specificity = np.array([x["Test_Specificity_macro"] for x in fold_metrics])

# ============================================================
# 29. 计算 5-Fold Mean ± SD
# ============================================================
print("\n")
print("=" * 70)
print("SVM Linear 5-Fold Cross-Validation Results")
print("=" * 70)
print(f"Accuracy    : {test_accuracy.mean():.4f} ± {test_accuracy.std(ddof=1):.4f}")
print(f"Precision   : {test_precision.mean():.4f} ± {test_precision.std(ddof=1):.4f}")
print(f"Recall      : {test_recall.mean():.4f} ± {test_recall.std(ddof=1):.4f}")
print(f"F1-score    : {test_f1.mean():.4f} ± {test_f1.std(ddof=1):.4f}")
print(f"Specificity : {test_specificity.mean():.4f} ± {test_specificity.std(ddof=1):.4f}")

# ============================================================
# 30. 输出每一 Fold 的 Macro 结果
# ============================================================
print("\n")
print("=" * 70)
print("Individual Fold Results")
print("=" * 70)
for result in fold_metrics:
    print(f"\nFold {result['Fold']}:")
    print(f"Accuracy    = {result['Test_Accuracy']:.4f}")
    print(f"Precision   = {result['Test_Precision_macro']:.4f}")
    print(f"Recall      = {result['Test_Recall_macro']:.4f}")
    print(f"F1-score    = {result['Test_F1_macro']:.4f}")
    print(f"Specificity = {result['Test_Specificity_macro']:.4f}")

# ============================================================
# 31. 输出每个类别的 5-Fold Mean ± SD
# ============================================================
print("\n")
print("=" * 100)
print("Per-Class Test Metrics: Mean ± SD")
print("=" * 100)
print(f"{'Class':<10}{'Precision':<22}{'Recall':<22}{'F1-score':<22}{'Specificity':<22}")
print("-" * 100)
for i, label in enumerate(labels):
    precision_mean = all_precision[:, i].mean()
    precision_sd = all_precision[:, i].std(ddof=1)
    recall_mean = all_recall[:, i].mean()
    recall_sd = all_recall[:, i].std(ddof=1)
    f1_mean = all_f1[:, i].mean()
    f1_sd = all_f1[:, i].std(ddof=1)
    specificity_mean = all_specificity[:, i].mean()
    specificity_sd = all_specificity[:, i].std(ddof=1)
    print(f"{str(label):<10}{precision_mean:.4f} ± {precision_sd:.4f}   {recall_mean:.4f} ± {recall_sd:.4f}   {f1_mean:.4f} ± {f1_sd:.4f}   {specificity_mean:.4f} ± {specificity_sd:.4f}")

# ============================================================
# 32. 保存 5-Fold Summary
# ============================================================
summary_metrics = {"Accuracy_mean": test_accuracy.mean(),
                   "Accuracy_SD": test_accuracy.std(ddof=1),
                   "Precision_macro_mean": test_precision.mean(),
                   "Precision_macro_SD": test_precision.std(ddof=1),
                   "Recall_macro_mean": test_recall.mean(),
                   "Recall_macro_SD": test_recall.std(ddof=1),
                   "F1_macro_mean": test_f1.mean(),
                   "F1_macro_SD": test_f1.std(ddof=1),
                   "Specificity_macro_mean": test_specificity.mean(),
                   "Specificity_macro_SD": test_specificity.std(ddof=1),
                   "Precision_per_class_mean": all_precision.mean(axis=0),
                   "Precision_per_class_SD": all_precision.std(axis=0, ddof=1),
                   "Recall_per_class_mean": all_recall.mean(axis=0),
                   "Recall_per_class_SD": all_recall.std(axis=0, ddof=1),
                   "F1_per_class_mean": all_f1.mean(axis=0),
                   "F1_per_class_SD": all_f1.std(axis=0, ddof=1),
                   "Specificity_per_class_mean": all_specificity.mean(axis=0),
                   "Specificity_per_class_SD": all_specificity.std(axis=0, ddof=1)}
joblib.dump(summary_metrics, os.path.join(outDir, "5fold_summary_metrics.pkl"))

# ============================================================
# 33. 保存每一 Fold 详细指标
# ============================================================
joblib.dump(fold_metrics, os.path.join(outDir, "5fold_all_metrics.pkl"))

# ============================================================
# 34. Overall Confusion Matrix
# ============================================================
overall_cm = confusion_matrix(all_test_true, all_test_pred, labels=labels)
print("\n")
print("=" * 70)
print("Overall Confusion Matrix")
print("=" * 70)
print(overall_cm)

# ============================================================
# 35. Overall 混淆矩阵转换为百分比
# ============================================================
overall_row_sums = overall_cm.sum(axis=1, keepdims=True)
overall_cm_percent = np.divide(overall_cm.astype(float), overall_row_sums, out=np.zeros_like(overall_cm, dtype=float), where=overall_row_sums != 0)
print("\n")
print("=" * 70)
print("Overall Confusion Matrix (%)")
print("=" * 70)
print(np.round(overall_cm_percent * 100, 2))

# ============================================================
# 36. Overall Per-Class Specificity
# ============================================================
overall_specificity_per_class = np.zeros(len(labels), dtype=float)
for i in range(len(labels)):
    tp = overall_cm[i, i]
    fp = np.sum(overall_cm[:, i]) - tp
    fn = np.sum(overall_cm[i, :]) - tp
    tn = np.sum(overall_cm) - tp - fp - fn
    if (tn + fp) > 0:
        overall_specificity_per_class[i] = tn / (tn + fp)
    else:
        overall_specificity_per_class[i] = 0.0

# ============================================================
# 37. Overall Macro-Specificity
# ============================================================
overall_specificity_macro = overall_specificity_per_class.mean()

# ============================================================
# 38. 输出 Overall Specificity
# ============================================================
print("\n")
print("=" * 70)
print("Overall Specificity")
print("=" * 70)
for i, label in enumerate(labels):
    print(f"Class {label} Specificity: {overall_specificity_per_class[i]:.4f}")
print(f"\nOverall Macro-Specificity: {overall_specificity_macro:.4f}")

# ============================================================
# 39. Overall Per-Class Precision / Recall / F1
# ============================================================
overall_precision_per_class = np.zeros(len(labels), dtype=float)
overall_recall_per_class = np.zeros(len(labels), dtype=float)
overall_f1_per_class = np.zeros(len(labels), dtype=float)
for i in range(len(labels)):
    tp = overall_cm[i, i]
    fp = np.sum(overall_cm[:, i]) - tp
    fn = np.sum(overall_cm[i, :]) - tp
    if (tp + fp) > 0:
        overall_precision_per_class[i] = tp / (tp + fp)
    if (tp + fn) > 0:
        overall_recall_per_class[i] = tp / (tp + fn)
    if (overall_precision_per_class[i] + overall_recall_per_class[i]) > 0:
        overall_f1_per_class[i] = 2 * overall_precision_per_class[i] * overall_recall_per_class[i] / (overall_precision_per_class[i] + overall_recall_per_class[i])

# ============================================================
# 40. 输出 Overall Per-Class Metrics
# ============================================================
print("\n")
print("=" * 100)
print("Overall Per-Class Metrics")
print("=" * 100)
print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'Specificity':<15}")
print("-" * 70)
for i, label in enumerate(labels):
    print(f"{str(label):<10}{overall_precision_per_class[i]:<15.4f}{overall_recall_per_class[i]:<15.4f}{overall_f1_per_class[i]:<15.4f}{overall_specificity_per_class[i]:<15.4f}")

# ============================================================
# 41. 保存 Overall 原始混淆矩阵
# ============================================================
joblib.dump(overall_cm, os.path.join(outDir, "overall_confusion_matrix.pkl"))

# ============================================================
# 42. 保存 Overall 百分比混淆矩阵
# ============================================================
joblib.dump(overall_cm_percent, os.path.join(outDir, "overall_confusion_matrix_percent.pkl"))

# ============================================================
# 43. 绘制 Overall 百分比混淆矩阵
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm_percent, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", values_format=".1%", colorbar=False)
ax.set_title("SVM Linear - 5-Fold Overall Confusion Matrix", fontname="Times New Roman", fontweight="bold", fontsize=14)
ax.set_xlabel("Predicted Label", fontname="Times New Roman", fontweight="bold", fontsize=14)
ax.set_ylabel("True Label", fontname="Times New Roman", fontweight="bold", fontsize=14)
for label in ax.get_xticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")
    label.set_fontsize(14)
for label in ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")
    label.set_fontsize(14)
for text in ax.texts:
    text.set_fontname("Times New Roman")
    text.set_fontweight("bold")
    text.set_fontsize(14)
plt.tight_layout()
plt.savefig(os.path.join(outDir, "SVM_Linear_5fold_overall_confusion_matrix_percent.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 44. 保存所有测试预测结果
# ============================================================
all_predictions = {"Y_test_all": all_test_true, "Y_pred_all": all_test_pred}
joblib.dump(all_predictions, os.path.join(outDir, "all_test_predictions_5fold.pkl"))

# ============================================================
# 45. 保存完整 CV Results
# ============================================================
cv_results = {"fold_metrics": fold_metrics,
              "precision_per_fold": all_precision,
              "recall_per_fold": all_recall,
              "f1_per_fold": all_f1,
              "specificity_per_fold": all_specificity,
              "specificity_macro_per_fold": test_specificity,
              "specificity_macro_mean": test_specificity.mean(),
              "specificity_macro_SD": test_specificity.std(ddof=1),
              "precision_per_class_mean": all_precision.mean(axis=0),
              "precision_per_class_SD": all_precision.std(axis=0, ddof=1),
              "recall_per_class_mean": all_recall.mean(axis=0),
              "recall_per_class_SD": all_recall.std(axis=0, ddof=1),
              "f1_per_class_mean": all_f1.mean(axis=0),
              "f1_per_class_SD": all_f1.std(axis=0, ddof=1),
              "specificity_per_class_mean": all_specificity.mean(axis=0),
              "specificity_per_class_SD": all_specificity.std(axis=0, ddof=1),
              "overall_confusion_matrix": overall_cm,
              "overall_confusion_matrix_percent": overall_cm_percent,
              "overall_precision_per_class": overall_precision_per_class,
              "overall_recall_per_class": overall_recall_per_class,
              "overall_f1_per_class": overall_f1_per_class,
              "overall_specificity_per_class": overall_specificity_per_class,
              "overall_specificity_macro": overall_specificity_macro,
              "Y_test_all": all_test_true,
              "Y_pred_all": all_test_pred,
              "labels": labels,
              "C": C}
joblib.dump(cv_results, os.path.join(outDir, "cv_results.pkl"))

# ============================================================
# 46. 完成
# ============================================================
print("\n")
print("=" * 70)
print("All SVM Linear 5-Fold results saved successfully.")
print("=" * 70)
print(f"\nOutput directory:\n{outDir}")