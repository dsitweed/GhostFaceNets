# import cv2
# import os

# def process_images(root_folder, output_folder):
#     # Lặp qua tất cả các thư mục con trong thư mục cha
#     for folder_name, subfolders, filenames in os.walk(root_folder):
#         # Tạo đường dẫn tương ứng trong thư mục output để lưu ảnh đã resize
#         folder_parent_name = os.path.basename(root_folder)
#         output_subfolder = os.path.join(output_folder, f"{folder_parent_name}_112_112", os.path.relpath(folder_name, root_folder))
#         os.makedirs(output_subfolder, exist_ok=True)

#         # Lặp qua tất cả các file ảnh trong thư mục hiện tại
#         for filename in filenames:
#             if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ xử lý các file ảnh
#                 input_path = os.path.join(folder_name, filename)

#                 # Đọc ảnh
#                 image = cv2.imread(input_path)

#                 if image is not None:
#                     # Resize ảnh về kích thước 112x112
#                     resized_image = cv2.resize(image, (112, 112))

#                     # Lưu ảnh đã resize vào thư mục mới
#                     output_filename = os.path.join(output_subfolder, filename)
#                     cv2.imwrite(output_filename, resized_image)

# # Thư mục cha chứa các thư mục con chứa ảnh cần xử lý
# root_folder = "./datasets/vn_celeb_imgs"
# # Thư mục để lưu các ảnh đã resize theo định dạng mong muốn
# output_folder = "./datasets/vn_celeb_small_112x112_folders"

# # Gọi hàm xử lý ảnh
# process_images(root_folder, output_folder)


# get embedding vector

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tqdm import tqdm
# from sklearn.preprocessing import normalize
# from sklearn.metrics import roc_curve, auc

# class Eval_image:
#     def __init__(self, model_interf) -> None:
#         if isinstance(model_interf, str) and model_interf.endswith(".h5"):
#             model = tf.keras.models.load_model(model_interf)
#             self.model = model
#             self.model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()


#     def prepare_image_and_embedding(self, img_path, output_dir=None):
#         img_shape = (112, 112)
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.expand_dims(img, axis=0)

#         emb = self.model_interf(img)
#         normalize(emb)

#         img_class = int(img_path.split('/')[-2])

#         return emb, img_class


# def run():
#     print("Hello")
#     model_path = 'checkpoints/ghostnetv1_w1.3_s2_basic_agedb_30_epoch_51_0.578333.h5'
#     single_img_path = 'datasets/bk_logs_celeb/1/00de8b4f-ff71-4b02-8444-020e112dfd8c_face.jpg'

#     eval_obj = Eval_image(model_path)

#     emb, img_class = eval_obj.prepare_image_and_embedding(single_img_path)
#     print("img class: ", img_class)


# run()

from matplotlib import pyplot as plt
import os
import zipfile
import cv2
import shutil

class FaceExtractor:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def extract_face(self, filename, required_size=(112, 112)):
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # detect faces in the image
        if len(faces) > 0:
            x1, y1, width, height = faces[0]
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]

            # resize pixels to the model size
            face_resized = cv2.resize(face, required_size)

            # Convert image to bytes
            _ , encoded_image = cv2.imencode('.jpg', face_resized)
            image_bytes = encoded_image.tobytes()
            return image_bytes
        else:
            return None


    def process_images(self, root_folder):
        extracted_images = {}

        for folder_name, subfolders, filenames in os.walk(root_folder):
            for folder_name_child, _, filenames_child in os.walk(folder_name):
                for filename in filenames_child:
                    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ xử lý các file ảnh
                        input_path = os.path.join(folder_name_child ,filename)
                        output_path = os.path.join(folder_name_child.split("/")[-2], filename)

                        extracted_image = self.extract_face(input_path)
                        if extracted_image:
                            extracted_images[output_path] = extracted_image


        return extracted_images

    def save_to_zip(self, extracted_images, zip_filename):
        with zipfile.ZipFile(zip_filename, 'w') as zip_file:
            for filename, image_bytes in extracted_images.items():
                zip_file.writestr(filename, image_bytes)

    def is_image_contain_face(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # detect faces in the image
        if len(faces) > 0:
            return True
        return False

def merge_images_to_folder(root_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for folder_name, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_file_path = os.path.join(folder_name, filename)

                destination_folder = os.path.join(target_folder, os.path.basename(folder_name))
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)

                destination_file_path = os.path.join(destination_folder ,filename)

                if os.path.exists(destination_file_path):
                    next

                shutil.copy(source_file_path, destination_file_path)


if __name__ == "__main__":
    ROOT_IMAGE_DIR = [
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_1',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_2',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_3',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_4',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_5',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_6',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_7',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_8',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_9',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_10',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_11',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_12',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_13',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_14',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_15',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_16',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_17',
        'datasets/bk_log_raw/50-001/cascade_extracted_face/extracted_faces_RK_test_18',
    ]

    face_extractor = FaceExtractor()
    # for root_dir in ROOT_IMAGE_DIR:
    #     ex_name = root_dir.split("/")[-1]
    #     extracted_images = face_extractor.process_images(root_folder=root_dir)
    #     face_extractor.save_to_zip(extracted_images, f'datasets/bk_log_raw/extracted_faces_{ex_name}.zip')
    #     print(f'FINISH save: extracted_faces_{ex_name}.zip')

    # Merged images
    merged_folder = 'datasets/bk_log_raw/50-001/cascade_extracted_face/merged_all'
    # for root_dir in ROOT_IMAGE_DIR:
    #     merge_images_to_folder(root_dir, 'datasets/bk_log_raw/50-001/cascade_extracted_face/merged_all')

    # Verify all image is contain face
    image_path = '/home/ky/NguyenVanKy/2023_2/GhostFaceNets/datasets/bk_log_raw/50-001/cascade_extracted_face/merged_all/190/4EV7T2T6_174018_full.jpg'

    # for folder_name, _, filenames in os.walk(merged_folder):
    #     for filename in filenames:
    #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             source_file_path = os.path.join(folder_name, filename)

    #             is_contain_face = face_extractor.is_image_contain_face(source_file_path)
    #             if not is_contain_face:
    #                 os.remove(source_file_path)

    for folder_name, subfolders, filenames in os.walk(merged_folder):
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_name, subfolder)
                subfolder_files = os.listdir(subfolder_path)
                if len(subfolder_files) < 3:
                    # Xóa thư mục con và tất cả các tập tin bên trong
                    shutil.rmtree(subfolder_path)
