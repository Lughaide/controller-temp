from .utils import *

# Pre + post processing for SSD_12 model

def preprocess_ssd(img: np.ndarray):
    img = cv2.resize(img, (1200, 1200), interpolation= cv2.INTER_LINEAR_EXACT)
    img = np.divide(img, 255.0)
    img = np.subtract(img, [0.485, 0.456, 0.406])
    img = np.divide(img, [0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1) # CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def reverse_ssd(img: np.ndarray):
    img = img.transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225]
    img = img + [0.485, 0.456, 0.406]
    img = img * 255.0
    return img

def postprocess_ssd(responses, model_outputs):
    cropped_result = {}
    
    for output_name in model_outputs:
        cropped_result[output_name] = []
    
    for response in responses:
        for output_name in model_outputs:
            for result in response.as_numpy(output_name):
                cropped_result[output_name].append(result)
    return cropped_result

# Pre + post processing for DenseNet model

def preprocess_dense(img: np.ndarray):
    img = cv2.resize(img, (224, 224), interpolation= cv2.INTER_LINEAR_EXACT)
    img = img.transpose(2, 0, 1) 
    img = img / 127.5 - 1 #type: ignore
    img = img.astype(np.float32)
    return img

def reverse_dense(img: np.ndarray):
    img = (img + 1) * 127.5
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    return img

def postprocess_dense(responses, output_name: str):
    output_array = responses.as_numpy(output_name)
    #print(output_array)
    # for results in output_array:
    #     for result in results:
    #         cls = result.split(':')
    #         #print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))
    #         print(cls)
    return output_array

def create_batch(img_dir: str, img_dim: tuple):
    img_batch = np.zeros(img_dim, dtype=np.float32)

    for img_name in glob.glob(f"{img_dir}/*.jpg"):
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img_batch = np.append(img_batch, preprocess_ssd(img), axis=0)
    return img_batch[1:]

def draw_img_w_label(img: np.ndarray, bbox: np.ndarray):
    color = (0, 0, 255)
    img_b = img.transpose(1, 2, 0)
    img_b = np.multiply(img_b, [0.229, 0.224, 0.225])
    img_b = np.add(img_b, [0.485, 0.456, 0.406])
    img_b = img_b * 255.0
    img_b = img_b.astype(np.uint8)
    bbox = bbox.astype(int)
    print(img_b.shape)
    start_point = (bbox[0], bbox[1])
    end_point = (bbox[2], bbox[3])
    img_c = cv2.rectangle(img_b.copy(), start_point, end_point, color, 4)
    cv2.imshow("Image with box", img_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return