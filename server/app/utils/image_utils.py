import cv2
import numpy as np
import io
import re
from skimage.transform import resize

def uploadedfile_to_cv2_image(uploaded_file):
    uploaded_file.seek(0)
    # Read the file data into a byte stream
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the byte stream to a cv2 image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    uploaded_file.seek(0)  # Move the file pointer back to the file's start in case you need to read it again
    return img

def cv2_image_to_byte_array(image):
    is_success, buffer = cv2.imencode(".png", image)
    io_buf = io.BytesIO(buffer)
    return io_buf

def read_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    # if is_bit_not:
    #     binary = cv2.bitwise_not(binary)
    #     image = cv2.bitwise_not(image)

    return image, binary

def extract_dimensions_from_path(path):
    # パス名から縦と横の長さを抽出
    match = re.search(r'(\d+)_(\d+)\.png$', path)
    if match:
        height = int(match.group(1))
        width = int(match.group(2))
        return height, width
    else:
        raise ValueError("パス名から縦横の長さを抽出できませんでした。")

def add_or_update_area(result_information_area_peri, index, area, non_area, perimeter, image_copy, mode, x, y, w, h):
    # 指定されたindexがリスト内に存在するか検索
    # mode 2は面積のみ減らす, 3が加算, 自信。
    found = False
    for item in result_information_area_peri:
        if item["index"] == index and mode == 3:
            # 全て足す場合
            item["area"] += area
            item["peri"] += perimeter
            found = True
            item["image"] = cv2.bitwise_xor(item["image"], image_copy)
            break
        elif item["index"] == index:
            item["area"] -= area
            item["peri"] += perimeter
            item["image"] = cv2.bitwise_xor(item["image"], image_copy)
            found = True
            break
    
    # 指定されたindexがリスト内に存在しない場合、新しい項目を追加
    if not found:
        if mode == 3:
            result_information_area_peri.append({
                "index": index,
                "area": area,
                "peri": perimeter,
                "image": image_copy,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })
        elif mode == 2:
            result_information_area_peri.append({
                "index": index,
                "area": -area,
                "peri": perimeter,
                "image": image_copy,
            })
    return result_information_area_peri
