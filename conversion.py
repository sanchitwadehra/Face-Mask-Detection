import tensorflowjs as tfjs

# Load the ml5.js model
model = ml5.neuralNetwork({
  model: 'models/200_epochs_850x850_64x64/model.json',
  metadata: 'models/200_epochs_850x850_64x64/model_meta.json',
  weights: 'models/200_epochs_850x850_64x64/model.weights.bin'
}, modelLoaded)

# Define the callback function to execute after the model is loaded
def modelLoaded():
    # Convert the ml5.js model to TensorFlow.js format
    tfjs.converters.save_keras_model(model.model, 'tfjs_model')
