from .utils import *

def preprocess_ssd(img: np.ndarray):
    img = cv2.resize(img, (1200, 1200), interpolation= cv2.INTER_LINEAR_EXACT)
    img = np.divide(img, 255.0)
    img = np.subtract(img, [0.485, 0.456, 0.406])
    img = np.divide(img, [0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def reverse_ssd(img: np.ndarray):
    img = img.transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225]
    img = img + [0.485, 0.456, 0.406]
    img = img * 255.0
    return img

def create_batch(img_dir: str):
    img_batch = np.zeros((1, 3, 1200, 1200), dtype=np.float32)

    for img_name in glob.glob(f"{img_dir}/*.jpg"):
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        img_batch = np.append(img_batch, preprocess_ssd(img), axis=0)
    return img_batch[1:]

def postprocess_ssd(img, responses, model_outputs):
    cropped_result = {}
    
    for output_name in model_outputs:
        cropped_result[output_name] = []
    
    for response in responses:
        for output_name in model_outputs:
            for result in response.as_numpy(output_name):
                cropped_result[output_name].append(result)
    return cropped_result

def postprocess_dense(responses, output_name: str):
    print(responses.get_response())
    # for response in responses:
    #     total_response = response.get_response()
    #     print(f"Response {total_response}")
    #     for result in response.as_numpy(output_name):
    #         pred = str(result, encoding='utf-8').split(":")
    #         print(pred)
    output_array = responses.as_numpy(output_name)
    print(output_array)
    for results in output_array:
        results = [results]
        for result in results:
            if output_array.dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                print(result)
                #cls = result.split(':')
                #print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))
    return

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