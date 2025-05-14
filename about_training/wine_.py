import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import pickle


# region 資料準備
wine_data = sklearn.datasets.load_wine(as_frame=True)

X = wine_data.data
y = wine_data.target
#print(X)
#print(y)
# endregion

# region 繪製HeatMap確認各Feature關係
temp_df = pd.concat([X,y],axis=1)
corr_temp_df = temp_df.corr()
plt.figure(figsize=(6,5))
plt.rcParams.update({'font.size':7})
sns.heatmap(corr_temp_df,annot=True,cmap='autumn')
#plt.xticks(rotation=50)
#plt.yticks(rotation=45)

plt.savefig("wine_feature_corr.png",bbox_inches='tight')
plt.show()
# endregion

# region 特徵篩選

# 具多元共線性的特徵對
# Total_phenols、flavonoids
# flavonoids、diluted_wines
X = X.drop(["flavanoids","od280/od315_of_diluted_wines"],axis = 1)
print("去除多元共線性問題後的Column：",X.columns)

# 使用SelectKBest選出與target有高線性相關的特徵
n = 5
selector = SelectKBest(f_classif,k = n)
selector.fit(X,y)
mask = selector.get_support()
new_cols = X.columns.values[mask]
X = X[new_cols]
print(f"SelectKBest 選擇前{n}特徵後的Column：",X.columns)
# endregion

# region 切分資料
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)
print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)
# endregion

# region 訓練模型
param_grid = {
    "kernel":["rbf","linear"],
    "C":[0.1,1,10],
    "gamma":["auto"]
}
svc_model = SVC()
grid = GridSearchCV(svc_model,param_grid,cv=5,scoring="accuracy",n_jobs=2)
grid.fit(train_X,train_y)
print("最佳參數組合：",grid.best_params_)
print("最佳模型評分(accuracy)：",grid.best_score_)
#print(grid.best_estimator_)

#之後pickle要放到models資料夾
#models_dir = ~/model-api-dir/models
with open("svc_model.pkl","wb") as file:
    pickle.dump(grid,file)
print("model saved！")
# endregion

#region 評估模型
from sklearn.metrics import accuracy_score
pred_y = grid.predict(test_X)
print("測試資料分數(accuracy)：",accuracy_score(test_y,pred_y))
# endregion