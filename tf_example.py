"""
Tensorflow example
"""
import akida
import keras
import numpy as np

from cnn2snn import convert
from keras.optimizers import Adam
from keras.datasets import mnist
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import set_akida_version, AkidaVersion

# TODO: Your design space
activation_bit = 4
weight_precision = 4
epochs = 1
learning_rate = 1e-3

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# TODO: Define your Keras Model
model_keras = keras.models.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(28, 28, 1)),
    
    keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same"),
    keras.layers.ReLU(max_value=2.0),

    keras.layers.Flatten(),
    keras.layers.Dense(10)
], 'mnistnet')

# Model summary: Print out the architecture of your model
model_keras.summary()

# Prepare for Training
model_keras.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['accuracy'])

# Train the model
print("\n[LOG]: Start Training...")
_ = model_keras.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

print("\n[LOG]: Evaluate the pre-trained model")
# Evaluate the model [TODO: Report this]
score = model_keras.evaluate(x_test, y_test, verbose=0)
print(f"\n[LOG] Test accuracy after pre-training: {score[1]}")

# Prepare the input samples for hardware 
xsamples = x_test
xsamples = xsamples[:,:,:,np.newaxis]
print(xsamples.shape)

# quantize the model (The input weight_bits is fixed = 8bit)
qparams = QuantizationParams(input_weight_bits=8, weight_bits=weight_precision, activation_bits=activation_bit)
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()

with set_akida_version(AkidaVersion.v1):
    # convert the CNN to SNN
    model_akida = convert(model_quantized)
    model_akida.summary()

    # Evaluate the accuracy of the converted model
    potentials_keras = model_quantized.predict(x_test, batch_size=100)
    preds_keras = np.squeeze(np.argmax(potentials_keras, 1))
    accuracy_keras = np.sum(np.equal(preds_keras, y_test)) / y_test.shape[0]

    devices = akida.devices()
    print(f'Available devices: {[dev.desc for dev in devices]}')
    device = devices[0]

    model_akida.map(device)
    model_akida.summary()

    device.soc.power_measurement_enabled = True

    _ = model_akida.forward(xsamples)

    floor_power = device.soc.power_meter.floor

    # Evaluate the hardware performance [TODO: Report this]
    print("\n============Hardware Measurement============")
    print(f"[MEASUREMENT] Floor power: {floor_power:.2f} mW")
    print(f"[MEASUREMENT] model akida power = {model_akida.statistics}")
    print(f"[MEASUREMENT] Total memory usage of the inference {device.memory[0]} Byte")
    print(f"[MEASUREMENT] Quantized model accuracy: {accuracy_keras*y_test.shape[0]:.0f}/{y_test.shape[0]}.")