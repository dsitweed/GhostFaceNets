import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import time
from tqdm import tqdm

def resize_and_save_image(image_path, target_path, new_size=(112, 112)):
  try:
    image = Image.open(image_path)

    transformer = transforms.Compose([
      transforms.Resize(new_size, transforms.InterpolationMode.NEAREST),
    ])

    resized_image = transformer(image)
    resized_image.save(target_path, format='JPEG')

  except Exception as e:
    print(f'Lỗi khi sử lý ảnh {image_path}: {e}')


def resize_and_save_all_folder(source_dir, target_dir):
  for root, dirs, files in os.walk(source_dir):
    for file in tqdm(files, desc="Resizing images"):
      if file.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(root, file)

        relative_path = os.path.relpath(image_path, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Add .jpg tail
        target_path = os.path.splitext(target_path)[0] + "_resized.jpg"

        resize_and_save_image(image_path=image_path, target_path=target_path)


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  source_dir = "datasets/bk_log_raw/final_use_folder_data/VN-celeb-big"
  target_dir = "datasets/bk_log_raw/final_use_folder_data/VN-celeb-big_112x112"

  os.makedirs(target_dir, exist_ok=True)
  new_size = (112, 112)

  print(device)

  start_time = time.time()

  resize_and_save_all_folder(source_dir, target_dir)

  print(f'Excuse time: {round(time.time() - start_time, 2)} s')
