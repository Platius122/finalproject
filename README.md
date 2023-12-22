# finalproject
    labels = [{'name':'metal', 'id':1}, {'name':'plastic', 'id':2}, {'name':'paper', 'id':3}]
model này em cho nhận diện 3 loại rác là lon nước, chai nhựa, với giấy
# Anh Hưng cứu emmm
1. Problem 1
em train xong model rồi, load xong luôn rồi, nhưng mà lúc chạy để detect thử thì nó báo lỗi từ cái cv2
![image](https://github.com/Platius122/finalproject/assets/146935747/a801cfa8-3d46-4a2e-b821-0aa8602da3e8)
em thử reinstall cái opecvn rồi nhưng vẫn lỗi
em đang sợ là do cái pip 23.3.2 tại nó auto update lên
em đang thử tạo một cái environment mới để chạy model thôi

đây là lần train thứ 2 rồi
lần 1 chạy được hết, nó nhận diện được luôn
lần 2 này nó gãy từ cái chỗ này

2. Problem 2
Em đang không cho vào Raspberry được

câu chuyện là cái tflite, anh thử install cái tflite trên máy tính rồi cho nó chạy thử ạ
tflite là cái folder em để trên đầu á, nó kiểu một model riêng nhưng chuyên chạy cho mấy cái máy nhỏ
lõi đa phần vẫn nằm ở cái bước opencv nó cứ xung đột với cái tensorflow, xung đột cả với python


3. problem 3
   ![image](https://github.com/Platius122/finalproject/assets/146935747/82d84a02-fd3f-4018-a13e-1426d9903aae)
   em cho chạy file tflite trong raspberry thì nó bị như này, bị lỗi ở biến count, lỗi chỉ nhận mảng 1 chiều mới chuyển thành int được
   em có thử đổi quá np.array thì lỗi là thành chỉ nhận int chứ không nhận array


# Load model
    import os
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    
    //Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    
    //Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-51')).expect_partial()
    
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
# Detect from an Image
    import cv2 
    import numpy as np
    from matplotlib import pyplot as plt
    %matplotlib inline
    
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', '<image_path>.jpg')
    
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()



# Real time detection from Webcam
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
