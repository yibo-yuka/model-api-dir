# 檔案說明

## wine_app.py

flask站台。

## about_training資料夾

wine_.py :訓練機器學習模型的過程。
wine_predictor.py：使用機器學習模型進行預測的範例。
wine_feature_corr.png：資料中各特徵、預測目標的相關係數熱力圖。
wine_data.xlsx：訓練資料，是scikit-learn提供的葡萄酒資料集。

## models資料夾

存放機器學習模型pickle的地方。

## templates資料夾

存放網頁畫面的地方。

## 這個Repository使用方法
這個專案在GCP VM上運行，並且使用Ubuntu 22.04 LST做為OS，所以可以使用linux terminal指令進行使用。
如果有需要使用的話，pull下來之後，
使用sudo gunicorn -w -2 -b 0.0.0.0:80 wine_app:app -daemon運行，即可使用以當前主機位址的80 port連到這個網頁。
如果是直接pull到電腦，就運行wine_app.py就好。
