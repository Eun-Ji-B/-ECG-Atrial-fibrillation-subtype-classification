import tensorflow as tf
#tensorflow 2.15.0
saved_model_dir = ".\\My_Model1126"

# Floating Point 32 Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()

with open('FP16_model1127.tflite', 'wb') as f:
  f.write(tflite_model)
#########################################################################################
# Floating Point 16 Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('FP16_model1127.tflite', 'wb') as f:
  f.write(tflite_model)
#########################################################################################
  
# 8-bit unsigned integer Convert the model
def representative_data(train_images):
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]
    
Non_saved_model_dir = "./1126test/Non"
PAF_saved_model_dir = "./1126test/PAF"
PsAF_saved_model_dir = "./1126test/PsAF"
print(representative_data(Non_saved_model_dir))

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('uint8_model1126.tflite', 'wb') as f:
  f.write(tflite_model)
  
