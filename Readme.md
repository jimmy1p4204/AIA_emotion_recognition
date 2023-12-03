訓練資料來源
- FER2013

訓練辨識模組的程式碼參考來源
- https://github.com/komalck/FACIAL-EMOTION-RECOGNITION/tree/master
- 因為參考程式碼語法過時，所以有找新的寫法
  - optimizer 新的 adam 寫法
     - https://keras-cn.readthedocs.io/en/latest/legacy/other/optimizers/
  - optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

訓練出來的模型:
- model.h5 (= model_Adam.h5)

如何使用情緒辨識功能
1. 照片取名 photo.png, 放到資料夾下
2. 執行 python main.py
