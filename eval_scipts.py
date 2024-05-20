import evals
from tensorflow import keras
import tensorflow as tf
import IJB_evals
import matplotlib.pyplot as plt
import keras_cv_attention_models
import GhostFaceNets, GhostFaceNets_with_Bias

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices('GPU'))

basic_model = keras.models.load_model('checkpoints/ghostnetv1_w1.3_s2_basic_lfw_epoch_2_0.572000.h5', compile=False)

ee = evals.eval_callback(basic_model, 'datasets/faces_emore/lfw.bin', batch_size=256, flip=True, PCA_acc=False)
ee.on_epoch_end(0)
