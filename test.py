import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_dir_size(dir_path):
    """Tính toán kích thước tổng của một thư mục."""
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            total_size += os.path.getsize(file_path)
    return total_size

def split_data_by_folder(source_dir, dest_dir, chunk_size):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    total_size = 0
    chunk_index = 0
    with tqdm(total=len(os.listdir(source_dir))) as pbar:
        for dir_name in os.listdir(source_dir):
            dir_path = os.path.join(source_dir, dir_name)
            if os.path.isdir(dir_path):
                dir_size = get_dir_size(dir_path)

                if total_size + dir_size > chunk_size:
                    chunk_index += 1
                    total_size = 0

                chunk_dir = os.path.join(dest_dir, f'glint360k_{chunk_index}')
                if not os.path.exists(chunk_dir):
                    os.makedirs(chunk_dir)

                shutil.copytree(dir_path, os.path.join(chunk_dir, dir_name))
                total_size += dir_size
            pbar.update(1)

def compress_chunk(chunk_path):
    zip_file = f"{chunk_path}.zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(chunk_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, chunk_path))
    return zip_file

def compress_chunks(chunk_dir):
    chunk_dirs = [os.path.join(chunk_dir, chunk) for chunk in os.listdir(chunk_dir)]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(compress_chunk, chunk): chunk for chunk in chunk_dirs}
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                pbar.update(1)

def verify_compression(chunk_dir):
    zip_files = [f for f in os.listdir(chunk_dir) if f.endswith('.zip')]
    for zip_file in zip_files:
        zip_path = os.path.join(chunk_dir, zip_file)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.testzip()
            print(f"{zip_file} is OK")
        except zipfile.BadZipFile:
            print(f"{zip_file} is corrupted")

if __name__ == "__main__":
    source_directory = '/u01/kynv/GhostFaceNets/datasets/glint360k/glint360k_112x112_folders'
    destination_directory = '/u01/kynv/GhostFaceNets/datasets/glint360k/glint360k_zip'
    chunk_size = 10 * 1024 * 1024 * 1024  # 10GB

    split_data_by_folder(source_directory, destination_directory, chunk_size)
    compress_chunks(destination_directory)
    verify_compression(destination_directory)
