import os
from tensorflow import keras
import losses, train, GhostFaceNets
import tensorflow as tf
import keras_cv_attention_models

if __name__ == "__main__":
  gpus = tf.config.experimental.list_physical_devices("GPU")
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  print(tf.config.list_physical_devices('GPU'))

  # Remove the below for better accuracies and keep it for faster training
  keras.mixed_precision.set_global_policy("mixed_float16")

  # (VN-Celeb) dataset
  data_path = 'datasets/vn_celeb_112x112_folders'
  eval_paths = ['datasets/faces_emore/lfw.bin', 'datasets/faces_emore/cfp_fp.bin', 'datasets/faces_emore/agedb_30.bin']

  # (Asian-Celeb) dataset
  data_path = 'datasets/small_asian_celeb_112x112_folders'
  data_path = 'datasets/asian_celeb_112x112_folders'
  eval_paths = ['datasets/faces_emore/lfw.bin']

  #GhostFaceNetV1
  # Strides of 2
  basic_model = GhostFaceNets.buildin_models("ghostnetv1_ky", dropout=0.2, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5, width=1)
  basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4, apply_to_batch_normal=False)
  basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)

  #Strides of 2
  tt = train.Train(data_path, eval_paths=eval_paths,
    save_path='ghostnetv1_w1.3_s2.h5',
    basic_model=basic_model, model=None, lr_base=0.1, lr_decay=0.5, lr_decay_steps=10, lr_min=1e-5,
    batch_size=16, random_status=0, eval_freq=1, output_weight_decay=1)

  # Train
  optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
  sch = [
      {"loss": losses.ArcfaceLoss(scale=32), "epoch": 1, "optimizer": optimizer},
      {"loss": losses.ArcfaceLoss(scale=64), "epoch": 1},
  ]
  tt.train(sch, 0)

  # Restore training from break point

  # tt = train.Train(data_path, 'ghostnetv1_w1.3_s2.h5', eval_paths, model='checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5',
  #               batch_size=64, random_status=0, lr_base=0.05, lr_decay=0.5, lr_decay_steps=35, lr_min=1e-5, eval_freq=1, output_weight_decay=1)

  # sch = [
  #   # {"loss": losses.ArcfaceLoss(scale=32), "epoch": 1, "optimizer": optimizer},
  #   {"loss": losses.ArcfaceLoss(scale=64), "epoch": 35},
  # ]

  # tt.train(sch, initial_epoch=15)
