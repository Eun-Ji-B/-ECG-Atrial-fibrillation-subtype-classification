# Tensorflow mnist 모델 코드
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

##데이터 세트 로드
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

##머신러닝 모델
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

##모델 훈련 및 평가
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

#model.export('export/my_model')
model.export('C:/Users/__/Desktop/labcop/code/My_Model')



#데이터 저장
'''
import tensorflow as tf
from PIL import Image

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

for i in range(10):
    img = (x_test[i] * 255).astype("uint8")
    label = int(y_test[i])
    Image.fromarray(img, mode="L").save(f"sample_{label}_{i}.png")



'''
