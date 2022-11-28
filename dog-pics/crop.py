import cv2
import glob

for img_name in glob.glob(f"./springer_*.jpg"):
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (1200, 1200), interpolation=cv2.INTER_LINEAR_EXACT)
    xmin, ymin, xmax, ymax = [174, 131, 1142, 1108]
    x, y, _ = img.shape
    cropped_img = img[ymin:ymax, xmin:xmax]
    cv2.imshow(f"{img_name}", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img.shape)