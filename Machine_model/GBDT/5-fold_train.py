import scipy.io as sio
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,log_loss
# ============================================================
# 1. 读取额外数据集
# ============================================================
data = sio.loadmat("../")
#data = sio.loadmat("../New_Data/104Peaks/test.mat")
X = data["X"]
Y = data["Y"].ravel()


# 用于保存外层5折及内层参数搜索的真实划分记录
split_records = []

print("=" * 60)
print("Dataset Information")
print("=" * 60)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
classes = np.unique(Y)
print("Classes:", classes)
print("\nClass distribution:")
for c in classes:
    print(f"Class {c}: {np.sum(Y == c)} samples")

# ============================================================
# 2. 5-Fold Stratified Cross Validation（外层CV评估模型性能）
# 每一轮：80% Training，20% Testing；共5轮，每一份数据都会作为一次测试集
# ============================================================
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 120],
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [3,5,7],
    "min_samples_leaf": [5,10,15]
}

# 内层3折分层交叉验证，在每折训练集上搜索最优超参
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ============================================================
# 4. 创建保存目录
# ============================================================
save_dir = "ML_Models/GBDT/folds"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# 5. 初始化结果
# ============================================================
# Overall metrics
acc_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_macro_list = []
# Per-class metrics
all_precision = []
all_recall = []
all_f1 = []
all_specificity = []
# 用于最终 pooled confusion matrix
all_y_true = []
all_y_pred = []
# 保存每折搜索得到的最优参数
best_params_list = []
# 保存每个 Fold 的 Training Loss 和 Test Loss
train_loss_all_folds = []
test_loss_all_folds = []
# ============================================================
# 6. 开始外层5-Fold
# ============================================================
for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, Y), start=1):

    print("\n")
    print("=" * 60)
    print(f"Fold {fold}")
    print("=" * 60)
    # ========================================================
    # 6.1 当前 Fold 的数据划分；train_idx = 80%，test_idx  = 20%
    # ========================================================
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples : {len(test_idx)}")

    # ========================================================
    # 6.2 GridSearchCV 在当前fold训练集上搜索最优参数
    # ========================================================
    base_gbdt = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=base_gbdt, param_grid=param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    print(f"Fold {fold} Best Params: {best_params}")
    print(f"Fold {fold} Inner CV best F1-macro: {grid_search.best_score_:.4f}")

    # ========================================================
    # 6.3 获取当前 Fold 最优模型的 Training Loss / Test Loss
    # ========================================================

    # --------------------------------------------------------
    # Training Loss
    # 每一轮 boosting iteration 的训练 loss
    # --------------------------------------------------------
    train_loss = best_model.train_score_

    # --------------------------------------------------------
    # Test Loss
    # 每一轮 boosting iteration 在外层测试集上的 Log Loss
    # --------------------------------------------------------
    test_loss = []

    for prob in best_model.staged_predict_proba(X_test):
        test_loss.append(
            log_loss(
                Y_test,
                prob,
                labels=classes
            )
        )

    test_loss = np.array(test_loss)
    # ========================================================
    # 找到当前 Fold 最优 Tree
    # 以外层 Test Loss 最低作为最优 Tree
    # ========================================================
    best_tree_idx = np.argmin(test_loss)
    best_tree = best_tree_idx + 1
    best_test_loss = test_loss[best_tree_idx]
    best_train_loss = train_loss[best_tree_idx]

    print(f"\nFold {fold} Best Tree:")
    print(f"Best Tree              : {best_tree}")
    print(f"Best Training Loss     : {best_train_loss:.6f}")
    print(f"Best Test Loss         : {best_test_loss:.6f}")
    # ========================================================
    # 保存当前 Fold 的 Loss 到 CSV
    # ========================================================
    loss_table = np.column_stack([
        np.arange(1, len(train_loss) + 1),
        train_loss,
        test_loss
    ])

    np.savetxt(
        f"{save_dir}/loss_fold_{fold}.csv",
        loss_table,
        delimiter=",",
        header="Iteration,Training_Loss,Test_Loss",
        comments=""
    )
    # 保存到所有 Fold
    train_loss_all_folds.append(train_loss)
    test_loss_all_folds.append(test_loss)

    # 打印当前 Fold Loss
    print(f"Boosting iterations : {len(train_loss)}")
    print(f"Final Training Loss : {train_loss[-1]:.6f}")
    print(f"Final Test Loss     : {test_loss[-1]:.6f}")

    # ========================================================
    # 使用搜索得到的最优模型，在当前 Fold 测试集上预测
    # ========================================================
    Y_pred = best_model.predict(X_test)

    # ========================================================
    # 6.5 保存当前 Fold 最优模型
    # ========================================================
    joblib.dump(best_model, f"{save_dir}/model_fold_{fold}.pkl")
    # ========================================================
    # 6.6 保存当前 Fold 的测试数据
    # ========================================================
    fold_test = {"X_test": X_test, "Y_test": Y_test, "test_idx": test_idx, "train_idx": train_idx}
    joblib.dump(fold_test, f"{save_dir}/test_{fold}.pkl")
    # ========================================================
    # 6.7 保存当前 Fold 的预测结果
    # ========================================================
    fold_prediction = {"Y_true": Y_test, "Y_pred": Y_pred, "test_idx": test_idx}
    joblib.dump(fold_prediction, f"{save_dir}/prediction_fold_{fold}.pkl")

    # ========================================================
    # 7. Overall ACC
    # ========================================================
    acc = accuracy_score(Y_test, Y_pred)
    # ========================================================
    # 8. Macro Precision / Recall / F1
    # ========================================================
    precision_macro = precision_score(Y_test, Y_pred, labels=classes, average="macro", zero_division=0)
    recall_macro = recall_score(Y_test, Y_pred, labels=classes, average="macro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred, labels=classes, average="macro", zero_division=0)
    # ========================================================
    # 9. Confusion Matrix
    # ========================================================
    cm = confusion_matrix(Y_test, Y_pred, labels=classes)
    # ========================================================
    # 打印当前 Fold 的混淆矩阵
    # ========================================================
    print(f"\nFold {fold} Confusion Matrix (raw count):")
    print(cm)
    cm_fold_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print(f"\nFold {fold} Confusion Matrix (row-normalized %):")
    print(np.round(cm_fold_percent * 100, 2))
    # ========================================================
    # 10. 每个类别 Specificity
    # ========================================================
    specificity = np.zeros(len(classes), dtype=float)
    for i in range(len(classes)):
        # True Negative
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        # False Positive
        fp = np.sum(cm[:, i]) - cm[i, i]
        if (tn + fp) == 0:
            specificity[i] = 0.0
        else:
            specificity[i] = tn / (tn + fp)
    # Macro specificity
    specificity_macro = np.mean(specificity)
    # ========================================================
    # 11. 每个类别 Precision
    # ========================================================
    precision_class = precision_score(Y_test, Y_pred, labels=classes, average=None, zero_division=0)
    # ========================================================
    # 12. 每个类别 Recall
    # ========================================================
    recall_class = recall_score(Y_test, Y_pred, labels=classes, average=None, zero_division=0)
    # ========================================================
    # 13. 每个类别 F1
    # ========================================================
    f1_class = f1_score(Y_test, Y_pred, labels=classes, average=None, zero_division=0)
    # ========================================================
    # 14. 保存当前 Fold 的 metrics
    # ========================================================
    acc_list.append(acc)
    precision_list.append(precision_macro)
    recall_list.append(recall_macro)
    f1_list.append(f1_macro)
    specificity_macro_list.append(specificity_macro)
    all_precision.append(precision_class)
    all_recall.append(recall_class)
    all_f1.append(f1_class)
    all_specificity.append(specificity)
    # ========================================================
    # 15. 保存 pooled predictions
    # ========================================================
    all_y_true.extend(Y_test)
    all_y_pred.extend(Y_pred)
    # ========================================================
    # 16. 打印当前 Fold 结果
    # ========================================================
    print("\nOverall:")
    print(f"ACC         : {acc:.4f}")
    print(f"Precision   : {precision_macro:.4f}")
    print(f"Recall      : {recall_macro:.4f}")
    print(f"F1          : {f1_macro:.4f}")
    print(f"Specificity : {specificity_macro:.4f}")
    print("\nPer-class results:")
    for i, c in enumerate(classes):
        print(f"Class {c}: Precision={precision_class[i]:.4f}, Recall={recall_class[i]:.4f}, F1={f1_class[i]:.4f}, Specificity={specificity[i]:.4f}")
# ============================================================
# 12. 绘制 5-Fold GBDT Training / Test Loss
# ============================================================

plt.figure(figsize=(8, 6))

for i in range(len(train_loss_all_folds)):

    train_loss = np.array(train_loss_all_folds[i])
    test_loss = np.array(test_loss_all_folds[i])

    iterations = np.arange(1, len(train_loss) + 1)

    plt.plot(
        iterations,
        train_loss,
        linewidth=1.8,
        label=f"Fold {i+1} Training Loss"
    )

    plt.plot(
        iterations,
        test_loss,
        linestyle="--",
        linewidth=1.8,
        label=f"Fold {i+1} Test Loss"
    )

plt.xlabel("Boosting Iteration", fontsize=14)
plt.ylabel("Log Loss", fontsize=14)

plt.title(
    "GBDT Training and Test Loss Across Boosting Iterations",
    fontsize=15
)

plt.legend(
    fontsize=9,
    frameon=True
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{save_dir}/GBDT_Training_Test_Loss_5Fold.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# 17. 转换为 numpy
# ============================================================
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)
all_f1 = np.array(all_f1)
all_specificity = np.array(all_specificity)

# ============================================================
# 18. 5-Fold Overall Mean ± SD
# ============================================================
print("\n")
print("=" * 60)
print("5-Fold Overall Results")
print("=" * 60)
print("ACC         : %.4f ± %.4f" % (np.mean(acc_list), np.std(acc_list)))
print("Precision   : %.4f ± %.4f" % (np.mean(precision_list), np.std(precision_list)))
print("Recall      : %.4f ± %.4f" % (np.mean(recall_list), np.std(recall_list)))
print("F1          : %.4f ± %.4f" % (np.mean(f1_list), np.std(f1_list)))
print("Specificity : %.4f ± %.4f" % (np.mean(specificity_macro_list), np.std(specificity_macro_list)))

# ============================================================
# 19. 5-Fold 每个类别 Mean ± SD
# ============================================================
print("\n")
print("=" * 60)
print("5-Fold Per-Class Results")
print("=" * 60)
for i, c in enumerate(classes):
    print(f"\nClass {c}")
    print("Precision   : %.4f ± %.4f" % (np.mean(all_precision[:, i]), np.std(all_precision[:, i])))
    print("Recall      : %.4f ± %.4f" % (np.mean(all_recall[:, i]), np.std(all_recall[:, i])))
    print("F1          : %.4f ± %.4f" % (np.mean(all_f1[:, i]), np.std(all_f1[:, i])))
    print("Specificity : %.4f ± %.4f" % (np.mean(all_specificity[:, i]), np.std(all_specificity[:, i])))

# ============================================================
# 20. Pooled Confusion Matrix
# 5个Fold的测试预测全部合并；每个样本只出现一次，因为每个样本只作为一次test
# ============================================================
cm_pooled = confusion_matrix(all_y_true, all_y_pred, labels=classes)
print("\n")
print("=" * 60)
print("Pooled Confusion Matrix")
print("=" * 60)
print(cm_pooled)

# ============================================================
# 21. 按真实类别进行行归一化
# ============================================================
cm_percent = cm_pooled.astype(float) / cm_pooled.sum(axis=1, keepdims=True)
print("\nPooled Confusion Matrix (%)")
print(np.round(cm_percent * 100, 2))

# ============================================================
# 22. 绘制 Pooled Confusion Matrix
# ============================================================
target_names = [str(c) for c in classes]
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=target_names)
disp.plot(cmap="Blues", values_format=".1%", ax=ax, colorbar=False)

# ============================================================
# 23. 标题
# ============================================================
ax.set_title("Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
ax.set_ylabel("True Label", fontsize=14, fontname="Times New Roman", fontweight="bold")

# ============================================================
# 24. 边框
# ============================================================
for spine in ax.spines.values():
    spine.set_linewidth(1)

# ============================================================
# 25. 坐标轴刻度
# ============================================================
ax.tick_params(axis="both", labelsize=12)
for label in ax.get_xticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")
for label in ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")

# ============================================================
# 26. 混淆矩阵数字
# ============================================================
for text in disp.text_.ravel():
    text.set_fontsize(14)
    text.set_fontname("Times New Roman")
    text.set_fontweight("bold")

plt.tight_layout()
plt.show()

# ============================================================
# 27. 保存所有结果
# ============================================================
cv_results = {
    # --------------------------------------------------------
    # Overall 5-fold results
    # --------------------------------------------------------
    "acc": np.array(acc_list),
    "precision_macro": np.array(precision_list),
    "recall_macro": np.array(recall_list),
    "f1_macro": np.array(f1_list),
    "specificity_macro": np.array(specificity_macro_list),
    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------
    "precision_per_class": all_precision,
    "recall_per_class": all_recall,
    "f1_per_class": all_f1,
    "specificity_per_class": all_specificity,
    # --------------------------------------------------------
    # Pooled predictions
    # --------------------------------------------------------
    "pooled_y_true": all_y_true,
    "pooled_y_pred": all_y_pred,
    # --------------------------------------------------------
    # Pooled confusion matrix
    # --------------------------------------------------------
    "pooled_confusion_matrix": cm_pooled,
    "pooled_confusion_matrix_percent": cm_percent,
    # --------------------------------------------------------
    # Class labels
    # --------------------------------------------------------
    "classes": classes,
    # --------------------------------------------------------
    # Each fold best hyperparameters
    # --------------------------------------------------------
    "best_params_all_fold": best_params_list,

    # --------------------------------------------------------
    # GBDT Training / Test Loss
    # --------------------------------------------------------
    "train_loss_all_folds": train_loss_all_folds,
    "test_loss_all_folds": test_loss_all_folds
}
joblib.dump(cv_results, f"{save_dir}/cv_results.pkl")

# ============================================================
# 28. 完成
# ============================================================
print("\n")
print("=" * 60)
print("5-Fold Cross Validation Completed")
print("=" * 60)
print(f"\nModels: {save_dir}/model_fold_1.pkl ~ model_fold_5.pkl")
print(f"Test sets: {save_dir}/test_1.pkl ~ test_5.pkl")
print(f"Predictions: {save_dir}/prediction_fold_1.pkl ~ prediction_fold_5.pkl")
print(f"Summary: {save_dir}/cv_results.pkl")
print("\nAll folds best parameters:")
for idx,p in enumerate(best_params_list,1):
    print(f"Fold{idx}: {p}")

# ============================================================
# 附加脚本：单独加载cv_results，重绘混淆矩阵（不需要重新跑CV）
# ============================================================
def plot_cm_from_result(result_path):
    res = joblib.load(result_path)
    cm_pooled = res["pooled_confusion_matrix"]
    cm_percent = res["pooled_confusion_matrix_percent"]
    classes = res["classes"]
    target_names = [str(c) for c in classes]
    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=target_names)

    disp.plot(cmap="Blues", values_format=".1%", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontname="Times New Roman", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
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
    plt.show()
# 使用示例：plot_cm_from_result(f"{save_dir}/cv_results.pkl")