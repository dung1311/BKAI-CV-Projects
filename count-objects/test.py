import cv2
import numpy as np

def preprocess_image1(image):
    """Xử lý ảnh 1 - Ảnh gốc chuẩn"""
    # Làm mờ nhẹ để giảm nhiễu nhỏ
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Phân ngưỡng Otsu
    _, binary = cv2.threshold(blurred, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Phép mở với kernel nhỏ
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opened

def preprocess_image2(image):
    """Xử lý ảnh 2 - Ảnh nhiễu màu"""
    # Chuyển về grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ mạnh để loại nhiễu điểm
    blurred = cv2.GaussianBlur(image, (7, 7), 2)
    
    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Phân ngưỡng thích ứng
    binary = cv2.adaptiveThreshold(enhanced, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 15, 5)
    
    # Phép mở với kernel lớn hơn để loại nhiễu
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opened

def preprocess_image3(image):
    """Xử lý ảnh 3 - Ảnh tối"""
    # Tăng độ sáng
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    
    # Cân bằng histogram
    equalized = cv2.equalizeHist(enhanced)
    
    # Làm mờ nhẹ
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Phân ngưỡng Otsu
    _, binary = cv2.threshold(blurred, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def preprocess_image4(image):
    """Xử lý ảnh 4 - Ảnh gradient"""
    # Áp dụng CLAHE để cân bằng độ sáng cục bộ
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Làm mờ để giảm gradient
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Phân ngưỡng thích ứng
    binary = cv2.adaptiveThreshold(blurred, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 5)
    
    # Phép mở để làm sạch nhiễu
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return opened

def count_objects(image, min_area=50, max_area=1000):
    """Đếm object từ ảnh đã xử lý"""
    # Tìm contour
    contours, _ = cv2.findContours(image, 
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contour theo diện tích
    valid_contours = [cnt for cnt in contours 
                     if min_area < cv2.contourArea(cnt) < max_area]
    
    return valid_contours

def process_and_display(image_path, image_type):
    """Xử lý và hiển thị kết quả"""
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Chọn phương pháp xử lý phù hợp
    if image_type == 1:
        processed = preprocess_image1(image)
        min_area, max_area = 50, 800
    elif image_type == 2:
        processed = preprocess_image2(image)
        min_area, max_area = 60, 800
    elif image_type == 3:
        processed = preprocess_image3(image)
        min_area, max_area = 40, 800
    else:  # image_type == 4
        processed = preprocess_image4(image)
        min_area, max_area = 50, 800
    
    # Đếm object
    valid_contours = count_objects(processed, min_area, max_area)
    
    # Vẽ kết quả
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, valid_contours, -1, (0,255,0), 1)
    
    # Đánh số
    for i, cnt in enumerate(valid_contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, str(i+1), (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    
    return len(valid_contours), result, processed

# Sử dụng cho từng ảnh
def process_all_images(image_paths):
    for i, path in enumerate(image_paths, 1):
        count, result, binary = process_and_display(path, i)
        
        # Hiển thị kết quả
        cv2.imshow(f'Ảnh {i} - Nhị phân', binary)
        cv2.imshow(f'Ảnh {i} - Kết quả đếm ({count} objects)', result)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def main():
    img_paths = ['./img1.png', './img2.png', './img3.png', './img4.png']
    process_all_images(img_paths)

if __name__ == '__main__':
    main()