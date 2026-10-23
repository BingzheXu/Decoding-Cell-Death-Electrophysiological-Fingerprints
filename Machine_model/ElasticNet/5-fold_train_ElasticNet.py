import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
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
outDir = "ML_Models/ElasticNet_5Fold"
os.makedirs(outDir, exist_ok=True)

# ============================================================
# 3. 外层 5-Fold Cross-Validation
# ============================================================
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 4. 参数搜索范围（可自行修改搜索网格）
# ============================================================
l1_ratios = np.linspace(0.1, 0.9, 9)  # ElasticNet l1_ratio
Cs = np.logspace(-3, 2, 10)           # 正则化系数C

# ============================================================
# 5. 保存结果
# ============================================================
fold_metrics = []
all_test_true = []
all_test_pred = []
confusion_matrices = []
best_params_list = []

labels = np.unique(Y)
# 用于存储每折各类指标
all_precision = []
all_recall = []
all_f1 = []
all_specificity = []

# ============================================================
# 6. 外层 5-Fold
# ============================================================
for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, Y), start=1):
    print("\n")
    print("=" * 70)
    print(f"Fold {fold}/5")
    print("=" * 70)
    # ========================================================
    # 6.1 当前 Fold 的 Train / Test
    # ========================================================
    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    # ========================================================
    # 6.2 当前 Fold 内部进行 5-fold 参数搜索
    # 注意：参数搜索只使用当前 Fold 的训练集；当前 Fold 的 test 数据完全不参与参数选择
    # ========================================================
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_acc = -np.inf
    best_params = None
    print("\nSearching best parameters...")
    for l1 in l1_ratios:
        for C in Cs:
            inner_fold_acc = []
            for inner_train_idx, inner_val_idx in inner_kf.split(X_train, Y_train):
                X_tr = X_train[inner_train_idx]
                X_val = X_train[inner_val_idx]
                Y_tr = Y_train[inner_train_idx]
                Y_val = Y_train[inner_val_idx]
                model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1, C=C, max_iter=5000, random_state=42)
                model.fit(X_tr, Y_tr)
                Y_val_pred = model.predict(X_val)
                acc = accuracy_score(Y_val, Y_val_pred)
                inner_fold_acc.append(acc)
            mean_acc = np.mean(inner_fold_acc)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = (l1, C)
    # ========================================================
    # 6.3 当前 Fold 最佳参数
    # ========================================================
    best_l1, best_C = best_params
    best_params_list.append({"l1_ratio": best_l1, "C": best_C})
    print("\nBest parameters for this Fold:")
    print(f"Best CV Accuracy : {best_acc:.4f}")
    print(f"Best l1_ratio    : {best_l1:.4f}")
    print(f"Best C           : {best_C:.6f}")
    # ========================================================
    # 6.4 使用当前 Fold 最佳参数，在整个当前训练集上重新训练
    # ========================================================
    final_model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=best_l1, C=best_C, max_iter=5000, random_state=42)
    final_model.fit(X_train, Y_train)
    print("\nFinal model training completed.")
    # ========================================================
    # 6.5 Training Set 预测
    # ========================================================
    train_pred = final_model.predict(X_train)
    # ========================================================
    # 6.6 Test Set 预测
    # ========================================================
    test_pred = final_model.predict(X_test)
    # ========================================================
    # 7. Training Set 指标
    # ========================================================
    train_acc = accuracy_score(Y_train, train_pred)
    train_precision = precision_score(Y_train, train_pred, average="macro", zero_division=0)
    train_recall = recall_score(Y_train, train_pred, average="macro", zero_division=0)
    train_f1 = f1_score(Y_train, train_pred, average="macro", zero_division=0)
    # ========================================================
    # 8. Test Set 指标
    # ========================================================
    test_acc = accuracy_score(Y_test, test_pred)
    test_precision = precision_score(Y_test, test_pred, average="macro", zero_division=0)
    test_recall = recall_score(Y_test, test_pred, average="macro", zero_division=0)
    test_f1 = f1_score(Y_test, test_pred, average="macro", zero_division=0)

    # 每个类别指标（和GBDT代码保持一致）
    cm = confusion_matrix(Y_test, test_pred, labels=labels)
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
        prec_per_class[i] = tp/(tp+fp) if (tp+fp)!=0 else 0
        rec_per_class[i] = tp/(tp+fn) if (tp+fn)!=0 else 0
        f1_per_class[i] = 2*prec_per_class[i]*rec_per_class[i]/(prec_per_class[i]+rec_per_class[i]) if (prec_per_class[i]+rec_per_class[i])!=0 else 0
        spec_per_class[i] = tn/(tn+fp) if (tn+fp)!=0 else 0
    all_precision.append(prec_per_class)
    all_recall.append(rec_per_class)
    all_f1.append(f1_per_class)
    all_specificity.append(spec_per_class)

    # ========================================================
    # 9. 输出当前 Fold 结果
    # ========================================================
    print("\n----------------------------------------")
    print(f"Fold {fold} Training Set:")
    print("----------------------------------------")
    print(f"Accuracy  : {train_acc:.4f}")
    print(f"Precision : {train_precision:.4f}")
    print(f"Recall    : {train_recall:.4f}")
    print(f"F1-score  : {train_f1:.4f}")
    print("\n----------------------------------------")
    print(f"Fold {fold} Test Set:")
    print("----------------------------------------")
    print(f"Accuracy  : {test_acc:.4f}")
    print(f"Precision : {test_precision:.4f}")
    print(f"Recall    : {test_recall:.4f}")
    print(f"F1-score  : {test_f1:.4f}")
    # ========================================================
    # 10. 保存当前 Fold 指标
    # ========================================================
    fold_result = {"Fold": fold, "Train_Accuracy": train_acc, "Train_Precision_macro": train_precision, "Train_Recall_macro": train_recall, "Train_F1_macro": train_f1, "Test_Accuracy": test_acc, "Test_Precision_macro": test_precision, "Test_Recall_macro": test_recall, "Test_F1_macro": test_f1, "Best_CV_Accuracy": best_acc, "Best_l1_ratio": best_l1, "Best_C": best_C}
    fold_metrics.append(fold_result)
    # ========================================================
    # 11. Test Set 混淆矩阵
    # ========================================================
    confusion_matrices.append(cm)
    print("\nTest Confusion Matrix (raw count):")
    print(cm)
    # ========================================================
    # 12. 保存当前 Fold 测试数据
    # ========================================================
    joblib.dump((X_test, Y_test), os.path.join(outDir, f"test_{fold}.pkl"))
    # ========================================================
    # 13. 保存当前 Fold 预测结果
    # ========================================================
    prediction_results = {"Y_test": Y_test, "Y_pred": test_pred, "test_indices": test_idx}
    joblib.dump(prediction_results, os.path.join(outDir, f"test_predictions_fold_{fold}.pkl"))
    # ========================================================
    # 14. 保存当前 Fold 指标
    # ========================================================
    fold_test_metrics = {"Accuracy": test_acc, "Precision_macro": test_precision, "Recall_macro": test_recall, "F1_macro": test_f1, "Best_CV_Accuracy": best_acc, "Best_l1_ratio": best_l1, "Best_C": best_C}
    joblib.dump(fold_test_metrics, os.path.join(outDir, f"test_metrics_fold_{fold}.pkl"))
    # ========================================================
    # 15. 保存当前 Fold 混淆矩阵
    # ========================================================
    joblib.dump(cm, os.path.join(outDir, f"confusion_matrix_fold_{fold}.pkl"))
    # ========================================================
    # 16. 绘制当前 Fold 混淆矩阵：行归一化百分比，不显示数字
    # ========================================================
    fig, ax = plt.subplots(figsize=(6,5))
    # 按行归一化（每行真实类别，百分比）
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="") # values_format="" 不写任何文字
    ax.set_title(f"ElasticNet - Fold {fold} Test Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontname="Times New Roman", fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontname="Times New Roman", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, f"ElasticNet_test_confusion_matrix_fold_{fold}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # ========================================================
    # 17. 保存当前 Fold 模型
    # ========================================================
    joblib.dump(final_model, os.path.join(outDir, f"ElasticNet_model_fold_{fold}.pkl"))
    # ========================================================
    # 18. 保存所有 Fold 的 Test Prediction
    # ========================================================
    all_test_true.extend(Y_test)
    all_test_pred.extend(test_pred)

# ============================================================
# 19. 转换为 numpy
# ============================================================
all_test_true = np.array(all_test_true)
all_test_pred = np.array(all_test_pred)
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)
all_f1 = np.array(all_f1)
all_specificity = np.array(all_specificity)

# ============================================================
# 20. 提取 5 个 Fold 的 Test 指标
# ============================================================
test_accuracy = np.array([x["Test_Accuracy"] for x in fold_metrics])
test_precision = np.array([x["Test_Precision_macro"] for x in fold_metrics])
test_recall = np.array([x["Test_Recall_macro"] for x in fold_metrics])
test_f1 = np.array([x["Test_F1_macro"] for x in fold_metrics])
test_specificity = np.mean(all_specificity, axis=1)
# ============================================================
# 21. 计算 Mean ± SD（ddof=1，样本标准差，论文标准）
# ============================================================
print("\n")
print("=" * 70)
print("ElasticNet 5-Fold Cross-Validation Results")
print("=" * 70)
print(f"Accuracy  : {test_accuracy.mean():.4f} ± {test_accuracy.std(ddof=1):.4f}")
print(f"Precision : {test_precision.mean():.4f} ± {test_precision.std(ddof=1):.4f}")
print(f"Recall    : {test_recall.mean():.4f} ± {test_recall.std(ddof=1):.4f}")
print(f"F1-score  : {test_f1.mean():.4f} ± {test_f1.std(ddof=1):.4f}")
print(f"Specificity : {test_specificity.mean():.4f} ± {test_specificity.std(ddof=1):.4f}")
# ============================================================
# 22. 输出每一 Fold 的结果
# ============================================================
print("\n")
print("=" * 70)
print("Individual Fold Results")
print("=" * 70)
for result in fold_metrics:
    print(f"\nFold {result['Fold']}:")
    print(f"Accuracy  = {result['Test_Accuracy']:.4f}")
    print(f"Precision = {result['Test_Precision_macro']:.4f}")
    print(f"Recall    = {result['Test_Recall_macro']:.4f}")
    print(f"F1-score  = {result['Test_F1_macro']:.4f}")
    print(f"l1_ratio  = {result['Best_l1_ratio']:.4f}")
    print(f"C         = {result['Best_C']:.6f}")

# ============================================================
# 23. 各类别指标（与GBDT代码完全对齐，逐类别mean±SD）
# ============================================================
print("\n")
print("=" * 70)
print("Per-class Metrics (Mean ± SD across 5 folds)")
print("=" * 70)
for i, cls in enumerate(labels):
    print(f"\nClass {cls}:")
    print(f"Precision: {all_precision[:,i].mean():.4f} ± {all_precision[:,i].std(ddof=1):.4f}")
    print(f"Recall   : {all_recall[:,i].mean():.4f} ± {all_recall[:,i].std(ddof=1):.4f}")
    print(f"F1       : {all_f1[:,i].mean():.4f} ± {all_f1[:,i].std(ddof=1):.4f}")
    print(f"Specificity: {all_specificity[:,i].mean():.4f} ± {all_specificity[:,i].std(ddof=1):.4f}")

# ============================================================
# 24. 保存 5-Fold Summary
# ============================================================
summary_metrics = {
    "Accuracy_mean": test_accuracy.mean(),
    "Accuracy_SD": test_accuracy.std(ddof=1),
    "Precision_macro_mean": test_precision.mean(),
    "Precision_macro_SD": test_precision.std(ddof=1),
    "Recall_macro_mean": test_recall.mean(),
    "Recall_macro_SD": test_recall.std(ddof=1),
    "F1_macro_mean": test_f1.mean(),
    "F1_macro_SD": test_f1.std(ddof=1),
    "PerClass_Precision": all_precision,
    "PerClass_Recall": all_recall,
    "PerClass_F1": all_f1,
    "PerClass_Specificity": all_specificity,
    "labels": labels,
    "best_params_all_fold": best_params_list
}
joblib.dump(summary_metrics, os.path.join(outDir, "5fold_summary_metrics.pkl"))

# ============================================================
# 25. 保存每一 Fold 详细指标
# ============================================================
joblib.dump(fold_metrics, os.path.join(outDir, "5fold_all_metrics.pkl"))
# ============================================================
# 26. 计算 Pooled Overall Confusion Matrix
# 每个样本只作为一次测试样本，因此所有5个Fold的预测可以直接合并
# ============================================================
cm_pooled = confusion_matrix(all_test_true, all_test_pred, labels=labels)
print("\n")
print("=" * 70)
print("Pooled Confusion Matrix (raw count)")
print("=" * 70)
print(cm_pooled)

# ============================================================
# 27. 按真实类别进行行归一化
# ============================================================
cm_percent = cm_pooled.astype(float) / cm_pooled.sum(axis=1, keepdims=True)
print("\nPooled Confusion Matrix (%)")
print(np.round(cm_percent * 100, 2))

# ============================================================
# 28. 绘制 Pooled Confusion Matrix
# ============================================================
target_names = [str(c) for c in labels]
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=target_names)
disp.plot(cmap="Blues", values_format=".1%", ax=ax, colorbar=False)

# ============================================================
# 29. 标题
# ============================================================
ax.set_title("Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
ax.set_ylabel("True Label", fontsize=14, fontname="Times New Roman", fontweight="bold")

# ============================================================
# 30. 边框
# ============================================================
for spine in ax.spines.values(): spine.set_linewidth(1)

# ============================================================
# 31. 坐标轴刻度
# ============================================================
ax.tick_params(axis="both", labelsize=12)
for label in ax.get_xticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold")
for label in ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold")

# ============================================================
# 32. 混淆矩阵单元格文字样式
# ============================================================
for text in disp.text_.ravel(): text.set_fontsize(14); text.set_fontname("Times New Roman"); text.set_fontweight("bold")
plt.tight_layout()

# ============================================================
# 33. 保存混淆矩阵图片
# ============================================================
plt.savefig(os.path.join(outDir, "ElasticNet_5fold_overall_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 34. 保存全部CV结果到cv_results.pkl
# ============================================================
cv_results = {"acc": test_accuracy, "precision_macro": test_precision, "recall_macro": test_recall, "f1_macro": test_f1, "acc_mean": test_accuracy.mean(), "acc_sd": test_accuracy.std(ddof=1), "precision_macro_mean": test_precision.mean(), "precision_macro_sd": test_precision.std(ddof=1), "recall_macro_mean": test_recall.mean(), "recall_macro_sd": test_recall.std(ddof=1), "f1_macro_mean": test_f1.mean(), "f1_macro_sd": test_f1.std(ddof=1), "precision_per_class": all_precision, "recall_per_class": all_recall, "f1_per_class": all_f1, "specificity_per_class": all_specificity, "pooled_y_true": all_test_true, "pooled_y_pred": all_test_pred, "pooled_confusion_matrix": cm_pooled, "pooled_confusion_matrix_percent": cm_percent, "classes": labels, "best_params_all_fold": best_params_list, "fold_metrics": fold_metrics}
joblib.dump(cv_results, os.path.join(outDir, "cv_results.pkl"))

# ============================================================
# 35. 单独保存混淆矩阵
# ============================================================
joblib.dump(cm_pooled, os.path.join(outDir, "overall_confusion_matrix.pkl"))
joblib.dump(cm_percent, os.path.join(outDir, "overall_confusion_matrix_percent.pkl"))

# ============================================================
# 36. 保存全部预测标签
# ============================================================
all_predictions = {"Y_test_all": all_test_true, "Y_pred_all": all_test_pred}
joblib.dump(all_predictions, os.path.join(outDir, "all_test_predictions_5fold.pkl"))

# ============================================================
# 37. 输出最终汇总信息
# ============================================================
print("\n")
print("=" * 70)
print("ElasticNet 5-Fold Cross Validation Completed")
print("=" * 70)
print(f"\nOutput directory:")
print(outDir)
print("\nOverall Metrics (Mean ± SD):")
print(f"Accuracy  : {test_accuracy.mean():.4f} ± {test_accuracy.std(ddof=1):.4f}")
print(f"Precision : {test_precision.mean():.4f} ± {test_precision.std(ddof=1):.4f}")
print(f"Recall    : {test_recall.mean():.4f} ± {test_recall.std(ddof=1):.4f}")
print(f"F1-score  : {test_f1.mean():.4f} ± {test_f1.std(ddof=1):.4f}")
print("\nAll folds best parameters:")
for idx, p in enumerate(best_params_list, 1): print(f"Fold{idx}: {p}")

# ============================================================
# 附加函数：加载cv_results.pkl，重绘Pooled混淆矩阵
# 无需重新运行5折交叉验证
# ============================================================
def plot_cm_from_result(result_path):
    res = joblib.load(result_path)
    cm_pooled = res["pooled_confusion_matrix"]
    cm_percent = res["pooled_confusion_matrix_percent"]
    classes = res["classes"]
    target_names = [str(c) for c in classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=target_names)
    disp.plot(cmap="Blues", values_format=".1%", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
    for spine in ax.spines.values(): spine.set_linewidth(1)
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold")
    for label in ax.get_yticklabels(): label.set_fontname("Times New Roman"); label.set_fontweight("bold")
    for text in disp.text_.ravel(): text.set_fontsize(14); text.set_fontname("Times New Roman"); text.set_fontweight("bold")
    plt.tight_layout()
    plt.show()

# 使用示例：
# plot_cm_from_result(os.path.join(outDir, "cv_results.pkl"))