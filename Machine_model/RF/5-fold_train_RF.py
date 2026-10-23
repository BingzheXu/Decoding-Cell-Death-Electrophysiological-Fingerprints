import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
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
outDir = "ML_Models/RF_5Fold"
os.makedirs(outDir, exist_ok=True)

# ============================================================
# 3. 5-Fold Stratified Cross-Validation
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 4. 随机森林参数
# ============================================================
n_estimators = 100
max_depth = None
min_samples_leaf = 2
max_features = "sqrt"
random_state = 42

# ============================================================
# 5. 保存每一折结果
# ============================================================
fold_metrics = []
all_test_true = []
all_test_pred = []
labels = np.unique(Y)
confusion_matrices = []
all_precision = []
all_recall = []
all_f1 = []
all_specificity = []

# ============================================================
# 6. 开始 5-Fold CV
# ============================================================
for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), start=1):
    print("\n")
    print("=" * 70)
    print(f"Fold {fold}/5")
    print("=" * 70)
    # 6.1 获取训练集和测试集
    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    # 6.2 建立随机森林
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1, random_state=random_state)
    # 6.3 训练模型
    rf_model.fit(X_train, Y_train)
    print("Random Forest training completed.")
    # 6.4 Training Set 预测
    train_pred = rf_model.predict(X_train)
    # 6.5 Test Set 预测
    test_pred = rf_model.predict(X_test)
    # 7. Training Set 指标
    train_acc = accuracy_score(Y_train, train_pred)
    train_precision = precision_score(Y_train, train_pred, average="macro", zero_division=0)
    train_recall = recall_score(Y_train, train_pred, average="macro", zero_division=0)
    train_f1 = f1_score(Y_train, train_pred, average="macro", zero_division=0)
    # 8. Test Set 指标
    test_acc = accuracy_score(Y_test, test_pred)
    test_precision = precision_score(Y_test, test_pred, average="macro", zero_division=0)
    test_recall = recall_score(Y_test, test_pred, average="macro", zero_division=0)
    test_f1 = f1_score(Y_test, test_pred, average="macro", zero_division=0)
    # 8.1 计算当前 Fold 混淆矩阵
    cm = confusion_matrix(Y_test, test_pred, labels=labels)
    # 8.2 计算每个类别的 Precision / Recall / F1 / Specificity
    n_class = len(labels)
    prec_per_class = np.zeros(n_class)
    rec_per_class = np.zeros(n_class)
    f1_per_class = np.zeros(n_class)
    spec_per_class = np.zeros(n_class)
    for i in range(n_class):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        if (tp + fp) != 0: prec_per_class[i] = tp/(tp+fp)
        else: prec_per_class[i] = 0
        if (tp + fn) != 0: rec_per_class[i] = tp/(tp+fn)
        else: rec_per_class[i] = 0
        if (prec_per_class[i] + rec_per_class[i]) != 0: f1_per_class[i] = 2*prec_per_class[i]*rec_per_class[i]/(prec_per_class[i]+rec_per_class[i])
        else: f1_per_class[i] = 0
        if (tn + fp) != 0: spec_per_class[i] = tn/(tn+fp)
        else: spec_per_class[i] = 0
    all_precision.append(prec_per_class)
    all_recall.append(rec_per_class)
    all_f1.append(f1_per_class)
    all_specificity.append(spec_per_class)
    test_specificity = np.mean(spec_per_class) # 当前 Fold 的 Macro-Specificity
    # 9. 输出当前 Fold 结果
    print("\n----------------------------------------")
    print(f"Fold {fold} Training Set:")
    print("----------------------------------------")
    print(f"Accuracy    : {train_acc:.4f}")
    print(f"Precision   : {train_precision:.4f}")
    print(f"Recall      : {train_recall:.4f}")
    print(f"F1-score    : {train_f1:.4f}")
    print("\n----------------------------------------")
    print(f"Fold {fold} Test Set:")
    print("----------------------------------------")
    print(f"Accuracy    : {test_acc:.4f}")
    print(f"Precision   : {test_precision:.4f}")
    print(f"Recall      : {test_recall:.4f}")
    print(f"F1-score    : {test_f1:.4f}")
    print(f"Specificity : {test_specificity:.4f}")
    # 10. 输出当前 Fold 每个类别的指标
    print("\nPer-class Metrics:")
    for i, cls in enumerate(labels):
        print(f"Class {cls}: Precision={prec_per_class[i]:.4f}, Recall={rec_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, Specificity={spec_per_class[i]:.4f}")
    # 11. 保存当前 Fold 指标
    fold_result = {"Fold": fold, "Train_Accuracy": train_acc, "Train_Precision_macro": train_precision, "Train_Recall_macro": train_recall, "Train_F1_macro": train_f1, "Test_Accuracy": test_acc, "Test_Precision_macro": test_precision, "Test_Recall_macro": test_recall, "Test_F1_macro": test_f1, "Test_Specificity_macro": test_specificity}
    fold_metrics.append(fold_result)
    # 12. 保存当前 Fold 混淆矩阵
    confusion_matrices.append(cm)
    print("\nTest Confusion Matrix (raw count):")
    print(cm)
    # 13. 保存当前 Fold 的测试数据
    joblib.dump((X_test, Y_test), os.path.join(outDir, f"test_{fold}.pkl"))
    # 14. 保存当前 Fold 的预测结果
    prediction_results = {"Y_test": Y_test, "Y_pred": test_pred, "test_indices": test_idx}
    joblib.dump(prediction_results, os.path.join(outDir, f"test_predictions_fold_{fold}.pkl"))
    # 15. 保存当前 Fold 指标
    fold_test_metrics = {"Accuracy": test_acc, "Precision_macro": test_precision, "Recall_macro": test_recall, "F1_macro": test_f1, "Specificity_macro": test_specificity}
    joblib.dump(fold_test_metrics, os.path.join(outDir, f"test_metrics_fold_{fold}.pkl"))
    # 16. 保存当前 Fold 混淆矩阵
    joblib.dump(cm, os.path.join(outDir, f"confusion_matrix_fold_{fold}.pkl"))
    # 17. 绘制当前 Fold 混淆矩阵，行归一化百分比
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format=".1%", colorbar=False)
    ax.set_title(f"Random Forest - Fold {fold} Test Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
    for spine in ax.spines.values(): spine.set_linewidth(1)
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontname("Times New Roman")
        text.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, f"RF_test_confusion_matrix_fold_{fold}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # 18. 保存当前 Fold 模型
    joblib.dump(rf_model, os.path.join(outDir, f"RF_model_fold_{fold}.pkl"))
    # 19. 保存所有 Fold 的 Test Prediction
    all_test_true.extend(Y_test)
    all_test_pred.extend(test_pred)

# ============================================================
# 20. 转换为 numpy array
# ============================================================
all_test_true = np.array(all_test_true)
all_test_pred = np.array(all_test_pred)
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)
all_f1 = np.array(all_f1)
all_specificity = np.array(all_specificity)

# ============================================================
# 21. 提取 5 个 Fold 的 Test 指标
# ============================================================
test_accuracy = np.array([x["Test_Accuracy"] for x in fold_metrics])
test_precision = np.array([x["Test_Precision_macro"] for x in fold_metrics])
test_recall = np.array([x["Test_Recall_macro"] for x in fold_metrics])
test_f1 = np.array([x["Test_F1_macro"] for x in fold_metrics])
test_specificity = np.array([x["Test_Specificity_macro"] for x in fold_metrics])

# ============================================================
# 22. 计算 5-Fold Mean ± SD
# ============================================================
print("\n")
print("=" * 70)
print("Random Forest 5-Fold Cross-Validation Results")
print("=" * 70)
print(f"Accuracy    : {test_accuracy.mean():.4f} ± {test_accuracy.std(ddof=1):.4f}")
print(f"Precision   : {test_precision.mean():.4f} ± {test_precision.std(ddof=1):.4f}")
print(f"Recall      : {test_recall.mean():.4f} ± {test_recall.std(ddof=1):.4f}")
print(f"F1-score    : {test_f1.mean():.4f} ± {test_f1.std(ddof=1):.4f}")
print(f"Specificity : {test_specificity.mean():.4f} ± {test_specificity.std(ddof=1):.4f}")

# ============================================================
# 23. 输出每一个 Fold 的结果
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
# 24. 各类别指标 Mean ± SD
# ============================================================
print("\n")
print("=" * 70)
print("Per-class Metrics (Mean ± SD across 5 folds)")
print("=" * 70)
for i, cls in enumerate(labels):
    print(f"\nClass {cls}:")
    print(f"Precision   : {all_precision[:, i].mean():.4f} ± {all_precision[:, i].std(ddof=1):.4f}")
    print(f"Recall      : {all_recall[:, i].mean():.4f} ± {all_recall[:, i].std(ddof=1):.4f}")
    print(f"F1          : {all_f1[:, i].mean():.4f} ± {all_f1[:, i].std(ddof=1):.4f}")
    print(f"Specificity : {all_specificity[:, i].mean():.4f} ± {all_specificity[:, i].std(ddof=1):.4f}")

# ============================================================
# 25. 保存 5-Fold Summary
# ============================================================
summary_metrics = {"Accuracy_mean": test_accuracy.mean(), "Accuracy_SD": test_accuracy.std(ddof=1), "Precision_macro_mean": test_precision.mean(), "Precision_macro_SD": test_precision.std(ddof=1), "Recall_macro_mean": test_recall.mean(), "Recall_macro_SD": test_recall.std(ddof=1), "F1_macro_mean": test_f1.mean(), "F1_macro_SD": test_f1.std(ddof=1), "Specificity_macro_mean": test_specificity.mean(), "Specificity_macro_SD": test_specificity.std(ddof=1), "PerClass_Precision": all_precision, "PerClass_Recall": all_recall, "PerClass_F1": all_f1, "PerClass_Specificity": all_specificity, "labels": labels}
joblib.dump(summary_metrics, os.path.join(outDir, "5fold_summary_metrics.pkl"))

# ============================================================
# 26. 保存每一 Fold 的详细指标
# ============================================================
joblib.dump(fold_metrics, os.path.join(outDir, "5fold_all_metrics.pkl"))

# ============================================================
# 27. Pooled Overall Confusion Matrix
# ============================================================
overall_cm = confusion_matrix(all_test_true, all_test_pred, labels=labels)
print("\n")
print("=" * 70)
print("Pooled Overall Confusion Matrix (Raw Count)")
print("=" * 70)
print(overall_cm)

# ============================================================
# 28. Pooled Confusion Matrix 行归一化百分比
# ============================================================
overall_cm_percent = overall_cm.astype(float) / overall_cm.sum(axis=1, keepdims=True)
print("\n")
print("=" * 70)
print("Pooled Overall Confusion Matrix (%)")
print("=" * 70)
print(np.round(overall_cm_percent * 100, 2))

# ============================================================
# 29. 保存 Overall Confusion Matrix
# ============================================================
joblib.dump(overall_cm, os.path.join(outDir, "overall_confusion_matrix.pkl"))
joblib.dump(overall_cm_percent, os.path.join(outDir, "overall_confusion_matrix_percent.pkl"))

# ============================================================
# 30. 绘制 Overall Confusion Matrix，行归一化百分比
# ============================================================
target_names = [str(c) for c in labels]
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=overall_cm_percent, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", values_format=".1%", colorbar=False)
ax.set_title("Random Forest - 5-Fold Overall Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
ax.set_ylabel("True Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
for spine in ax.spines.values(): spine.set_linewidth(1)
ax.tick_params(axis="both", labelsize=12)
for label in ax.get_xticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")
for text in disp.text_.ravel():
    text.set_fontsize(14)
    text.set_fontname("Times New Roman")
    text.set_fontweight("bold")
plt.tight_layout()
plt.savefig(os.path.join(outDir, "RF_5fold_overall_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 31. 保存所有 Fold 的测试预测结果
# ============================================================
all_predictions = {"Y_test_all": all_test_true, "Y_pred_all": all_test_pred}
joblib.dump(all_predictions, os.path.join(outDir, "all_test_predictions_5fold.pkl"))

# ============================================================
# 32. 保存完整 CV 结果
# ============================================================
cv_results = {"acc": test_accuracy, "precision_macro": test_precision, "recall_macro": test_recall, "f1_macro": test_f1, "specificity_macro": test_specificity, "acc_mean": test_accuracy.mean(), "acc_sd": test_accuracy.std(ddof=1), "precision_macro_mean": test_precision.mean(), "precision_macro_sd": test_precision.std(ddof=1), "recall_macro_mean": test_recall.mean(), "recall_macro_sd": test_recall.std(ddof=1), "f1_macro_mean": test_f1.mean(), "f1_macro_sd": test_f1.std(ddof=1), "specificity_macro_mean": test_specificity.mean(), "specificity_macro_sd": test_specificity.std(ddof=1), "precision_per_class": all_precision, "recall_per_class": all_recall, "f1_per_class": all_f1, "specificity_per_class": all_specificity, "pooled_y_true": all_test_true, "pooled_y_pred": all_test_pred, "pooled_confusion_matrix": overall_cm, "pooled_confusion_matrix_percent": overall_cm_percent, "classes": labels, "fold_metrics": fold_metrics}
joblib.dump(cv_results, os.path.join(outDir, "cv_results.pkl"))

# ============================================================
# 33. 最终汇总输出
# ============================================================
print("\n")
print("=" * 70)
print("Random Forest 5-Fold Cross Validation Completed")
print("=" * 70)
print("\nOutput directory:")
print(outDir)
print("\nOverall Metrics (Mean ± SD):")
print(f"Accuracy    : {test_accuracy.mean():.4f} ± {test_accuracy.std(ddof=1):.4f}")
print(f"Precision   : {test_precision.mean():.4f} ± {test_precision.std(ddof=1):.4f}")
print(f"Recall      : {test_recall.mean():.4f} ± {test_recall.std(ddof=1):.4f}")
print(f"F1-score    : {test_f1.mean():.4f} ± {test_f1.std(ddof=1):.4f}")
print(f"Specificity : {test_specificity.mean():.4f} ± {test_specificity.std(ddof=1):.4f}")

# ============================================================
# 34. 输出每个类别的最终指标
# ============================================================
print("\n")
print("=" * 70)
print("Final Per-class Metrics")
print("=" * 70)
for i, cls in enumerate(labels):
    print(f"\nClass {cls}:")
    print(f"Precision   : {all_precision[:, i].mean():.4f} ± {all_precision[:, i].std(ddof=1):.4f}")
    print(f"Recall      : {all_recall[:, i].mean():.4f} ± {all_recall[:, i].std(ddof=1):.4f}")
    print(f"F1          : {all_f1[:, i].mean():.4f} ± {all_f1[:, i].std(ddof=1):.4f}")
    print(f"Specificity : {all_specificity[:, i].mean():.4f} ± {all_specificity[:, i].std(ddof=1):.4f}")

# ============================================================
# 35. 完成
# ============================================================
print("\n")
print("=" * 70)
print("All 5-Fold results saved successfully.")
print("=" * 70)
print("\nOutput directory:")
print(outDir)