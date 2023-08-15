import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

def extract_features(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (128, 128))
    
    features = []
    for channel in range(3):  # RGB channels
        pixels = image[:, :, channel].ravel()
        features.extend([np.mean(pixels), np.std(pixels), np.median(pixels),
                         skew(pixels), kurtosis(pixels), np.mean(-pixels * np.log2(pixels + np.finfo(float).eps))])
    return features


train_df = pd.read_csv('/data/gongmo/team1/gongmo_2023/csv/ddpm_for_128_10000/train_fold1.csv')
X_train = []
y_train = []

for index, row in train_df.iterrows():
    features = extract_features(row['filepath'])
    X_train.append(features)
    y_train.append(row['label'])

X_train = np.array(X_train)
y_train = np.array(y_train)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)


test_df = pd.read_csv('/data/gongmo/team1/gongmo_2023/csv/ddpm_for_128_10000/test_fold0.csv')
X_test = []
y_test = []

for index, row in test_df.iterrows():
    features = extract_features(row['filepath'])
    X_test.append(features)
    y_test.append(row['label'])

X_test = np.array(X_test)
y_test = np.array(y_test)



y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Confusion Matrix를 계산합니다.
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion Matrix의 각 요소를 가져옵니다.
tn, fp, fn, tp = conf_matrix.ravel()

# Sensitivity와 Specificity를 계산합니다.
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# F1 Score를 계산합니다.
f1 = f1_score(y_test, y_pred)

# 결과를 출력합니다.
print("Confusion Matrix:")
print(conf_matrix)
print("\nSensitivity:", sensitivity)
print("Specificity:", specificity)
print("F1 Score:", f1)


y_prob = clf.predict_proba(X_test)[:, 1]  # Positive class에 대한 예측 확률
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# ROC curve를 그립니다.
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("AUC Score:", auc_score)