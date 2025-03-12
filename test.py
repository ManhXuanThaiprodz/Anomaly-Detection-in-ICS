import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Tạo dữ liệu giả lập
np.random.seed(42)
ytrue = np.random.randint(0, 2, 100)  # Giá trị thực tế (0 hoặc 1)
ypred_probs = np.random.rand(100)  # Xác suất dự đoán của mô hình

# Tính TPR và FPR
fpr, tpr, _ = roc_curve(ytrue, ypred_probs)
roc_auc = auc(fpr, tpr)  # Tính diện tích AUC

# Vẽ biểu đồ ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Đường chéo ngẫu nhiên
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid()
# plt.show()
plt.savefig("roc_curve.png")  # Lưu file
print("ROC Curve đã được lưu vào roc_curve.png")

