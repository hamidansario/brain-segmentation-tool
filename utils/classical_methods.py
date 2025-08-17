import cv2

def canny_edge_detection(image_path, low_threshold, high_threshold):
    img = cv2.imread(image_path, 0)  # Read as grayscale
    edges = cv2.Canny(img, low_threshold, high_threshold)
    output_path = image_path.rsplit('.', 1)[0] + '_canny.png'
    cv2.imwrite(output_path, edges)
    return output_path

def otsu_thresholding(image_path):
    img = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output_path = image_path.rsplit('.', 1)[0] + '_otsu.png'
    cv2.imwrite(output_path, thresh)
    return output_path
