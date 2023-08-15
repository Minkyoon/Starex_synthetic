import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import xgboost as xgb
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import canny


for i in range(5):

    num=i

    

    def extract_features(img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (32, 32))
        
        features = []
        
        # RGB Channel Features
        for channel in range(3):
            pixels = image[:, :, channel].ravel()
            features.extend([np.mean(pixels), np.std(pixels), np.median(pixels),
                            skew(pixels), kurtosis(pixels), np.mean(-pixels * np.log2(pixels + np.finfo(float).eps))])
        
        # Grayscale conversion for certain features
        gray_image = rgb2gray(image)
        
        # LBP (Local Binary Pattern) Features
        radius = 1  # LBP radius
        n_points = 8 * radius  # Number of points to be considered as neighbours 
        lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        features.extend(hist)
        
        # Haralick Texture Features
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix((gray_image * 255).astype(np.uint8), distances=distances, angles=angles, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        haralick_features = [contrast, dissimilarity, homogeneity, energy, correlation]
        for feature in haralick_features:
            features.extend(np.ravel(feature))
        
        # Histogram Features
        hist_features = np.histogram(image, bins=256)[0]
        features.extend(hist_features)
        
        # Canny Edge Features
        edges = canny(gray_image)
        edge_count = np.sum(edges)
        features.append(edge_count)
        fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(2, 2),
                        cells_per_block=(1, 1), visualize=True,)
        features.extend(fd)

        # HSV Color Space Features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        features.extend([h.mean(), s.mean(), v.mean(), h.std(), s.std(), v.std()])
        
            
        return features


    train_df = pd.read_csv(f'/data/gongmo/team1/gongmo_2023/csv/externel_5fold/train_fold{num}.csv')
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

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist')
    clf.fit(X_train, y_train)


    test_df = pd.read_csv(f'/data/gongmo/team1/gongmo_2023/csv/externel_5fold/test_fold{num}.csv')
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



# Confusion Matrix를 그립니다.
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{z}', ha='center', va='center', color='black', fontsize=15)

    plt.title('Confusion matrix', pad=20)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_fold{num}.png')



    # 지표들을 TXT 파일에 저장합니다.
    with open(f'metrics_fold{num}.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"AUC Score: {auc_score}\n")