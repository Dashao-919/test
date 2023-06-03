import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# 将像素值缩放到 0 到 1 的范围内
x_test = x_test / 255.0

# 加载模型
model = tf.keras.models.load_model('shenduxx.h5')

# 进行预测并可视化测试样本和预测结果
def predict(image):
    predictions = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(predictions[0])
    return predicted_label

# 显示 10 个随机测试样本
indices = np.random.randint(0, x_test.shape[0], size=10)
for i, index in enumerate(indices):
    image = x_test[index]
    true_label = y_test[index]
    predicted_label = predict(image)

    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title('True label: {}, Predicted label: {}'.format(true_label, predicted_label))
    plt.axis('off')
plt.show()

# 加载并预测自己的图像
pil_image = Image.open("test.png").convert('L')
pil_image = pil_image.resize((28, 28))
image = np.array(pil_image)
# image = image / 255.0
# image = image.reshape(1, 28*28)
image = np.invert(image)
predicted_label = predict(image)
plt.imshow(image, cmap=plt.cm.binary)
plt.title('Predicted label: {}'.format(predicted_label))
plt.axis('off')
plt.show()
