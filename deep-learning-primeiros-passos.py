# 1 = Importe as bibliotecas("pip install tensorflow, keras")
import tensorflow as tf
import keras
# 2 = Adquira Dados
(train_images, train_labels), (imagens_teste, categorias_teste) = keras.datasets.mnist.load_data()
# 3 = Setar Modelo Preditivo
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# 4 = Treino
model.fit(train_images,train_labels, epochs=10)
# 5 = Avaliação
loss, acc = model.evaluate(imagens_teste,categorias_teste)
print(f'Acurácia no teste: {acc}')
# 6 = Predições
pred = model.predict(imagens_teste)