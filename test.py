import cv2
import os

def process_images(root_folder, output_folder):
    # Lặp qua tất cả các thư mục con trong thư mục cha
    for folder_name, subfolders, filenames in os.walk(root_folder):
        # Tạo đường dẫn tương ứng trong thư mục output để lưu ảnh đã resize
        folder_parent_name = os.path.basename(root_folder)
        output_subfolder = os.path.join(output_folder, f"{folder_parent_name}_112_112", os.path.relpath(folder_name, root_folder))
        os.makedirs(output_subfolder, exist_ok=True)

        # Lặp qua tất cả các file ảnh trong thư mục hiện tại
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ xử lý các file ảnh
                input_path = os.path.join(folder_name, filename)

                # Đọc ảnh
                image = cv2.imread(input_path)

                if image is not None:
                    # Resize ảnh về kích thước 112x112
                    resized_image = cv2.resize(image, (112, 112))

                    # Lưu ảnh đã resize vào thư mục mới
                    output_filename = os.path.join(output_subfolder, filename)
                    cv2.imwrite(output_filename, resized_image)

# Thư mục cha chứa các thư mục con chứa ảnh cần xử lý
root_folder = "./datasets/vn_celeb_imgs"
# Thư mục để lưu các ảnh đã resize theo định dạng mong muốn
output_folder = "./datasets/vn_celeb_small_112x112_folders"

# Gọi hàm xử lý ảnh
process_images(root_folder, output_folder)
