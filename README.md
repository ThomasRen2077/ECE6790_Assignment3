# ECE6790_Assignment3

Activate the installed environment:

```bash
conda activate akida
```

## Dataset

- **Dataset:** MNIST dataset with the input size = [28, 28]

## Coding

Create a keras model by customizing the `model_keras`

```python
model_keras = keras.models.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(28, 28, 1)),

    keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same"),
    keras.layers.ReLU(max_value=3.0),
    keras.layers.MaxPooling2D((2, 2), padding="same"),

    keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same"),
    keras.layers.ReLU(max_value=3.0),
    keras.layers.MaxPooling2D((2, 2), padding="same"),

    keras.layers.Flatten(),
    keras.layers.Dense(10)
], 'mnistnet')
```

Quantize the model with the designed precision (with the default quantizer)

```python
# quantize the model (The input weight_bits is fixed = 8bit)
qparams = QuantizationParams(input_weight_bits=8, weight_bits=weight_precision, activation_bits=activation_bit)
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()
```

Measure the power and latency (frame per second) by running model inference on hardware.

- Convert the quantized model to spiking neural network

  ```python
  with set_akida_version(AkidaVersion.v1):
   		# convert the CNN to SNN
      model_akida = convert(model_quantized)
      model_akida.summary()
  ```

- Evaluate the accuracy of the converted model:

  ```python
  # Evaluate the accuracy of the converted model
  potentials_keras = model_quantized.predict(x_test, batch_size=100)
  preds_keras = np.squeeze(np.argmax(potentials_keras, 1))
  accuracy_keras = np.sum(np.equal(preds_keras, y_test)) / y_test.shape[0]
  ```

- Deploy the model

  ```python
  model_akida.map(device)
  model_akida.summary()

  device.soc.power_measurement_enabled = True
  ```

- The hardware performance is measured based on all the 10,000 samples:

  ```python
  # Prepare the input samples for hardware
  xsamples = x_test
  xsamples = xsamples[:,:,:,np.newaxis]
  print(xsamples.shape)
  ```

- The hardware performance is measured by Akida tool, which returns the latency, power, and energy consumption:

## Report

- Accuracy:

```bash
Test accuracy after pre-training: 0.9872999787330627
Quantized model accuracy: 9761/10000.
```

- Overall Performance:

```bash
Floor power: 894.26 mW
model akida power = Average framerate = 1677.85 fps
LLast inference power range (mW):  Avg 895.27 / Min 894.00 / Max 907.00 / Std 2.93
Last inference energy consumed (mJ/frame): 0.53
Total memory usage of the inference 96304 Byte
Quantized model accuracy: 9761/10000.
```
