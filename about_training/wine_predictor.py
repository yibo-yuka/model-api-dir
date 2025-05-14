import pickle
import numpy as np

# 'alcohol', 'total_phenols', 'color_intensity', 'hue', 'proline'
input_value = np.array([[12,1.45,3.6,1.05,450]])

with open("svc_model.pkl","rb") as file:
    svc_model = pickle.load(file)

pred_result = svc_model.predict(input_value)
print(pred_result)