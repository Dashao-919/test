import tensorflow as tf
from tensorflow.keras import layers, models

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将像素值缩放到 0 到 1 的范围内
x_train, x_test = x_train / 255.0, x_test / 255.0

# 定义模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10)
])

# 编译和训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.save('shenduxx.h5')

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# 进行预测
predictions = model.predict(x_test)
print('Predictions:', predictions[0])
print('Class with highest probability:', tf.argmax(predictions[0]))
print('True label:', y_test[0])
