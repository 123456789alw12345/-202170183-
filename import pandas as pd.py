import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# 1. تحميل بيانات وتحليلها باستخدام Pandas
# إنشاء DataFrame عشوائي يمثل مجموعة بيانات
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.randint(1, 100, 100),
    'feature2': np.random.randint(1, 100, 100),
    'label': np.random.choice([0, 1], 100)
})

print("معاينة البيانات:")
print(data.head())

# إحصاءات وصفية للبيانات
print("\nالإحصاءات الوصفية:")
print(data.describe())

# 2. رسم بياني باستخدام Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(data['feature1'], data['feature2'], c=data['label'], cmap='coolwarm')
plt.title('رسم بياني لمتغيرات البيانات')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.colorbar(label='label')
plt.show()

# 3. تدريب نموذج تعلم آلي باستخدام Scikit-learn
# فصل البيانات إلى مدخلات ومخرجات
X = data[['feature1', 'feature2']]
y = data['label']

# تقسيم البيانات إلى بيانات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج التصنيف باستخدام RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ بالبيانات الاختبارية
y_pred = model.predict(X_test)

# حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"\nدقة النموذج: {accuracy * 100:.2f}%")

# 4. خوارزمية البحث الثنائي
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# قائمة مرتبة للبحث
arr = np.sort(np.random.randint(1, 100, 20))
target = 50

start_time = time.time()
result = binary_search(arr, target)
end_time = time.time()

if result != -1:
    print(f"\nالعنصر {target} موجود في الموقع {result}.")
else:
    print(f"\nالعنصر {target} غير موجود في القائمة.")

print(f"زمن تنفيذ البحث: {end_time - start_time:.6f} ثانية")
