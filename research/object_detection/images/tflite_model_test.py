import numpy as np
import tensorflow as tf
import cv2  # 用来读取图片并进行预处理
import glob  # 读取某文件夹所有测试图片
import time  # 主要是用来计算推理花费时间

# Load TFLite model and allocate tensors.
model_path = "./ckpt/output.tflite"  # tflite路径
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)  # 在这里可以看到tflite的输入输出的节点信息


def detection(img_src):
    img = cv2.resize(img_src, (300, 300))
    img = img / 128 - 1
    input_data = np.expand_dims(img, 0)
    input_data = input_data.astype(np.float32)
    # 以上是对图片经行尺寸变换、归一化、添加维度和类型转换，以便和输入节点对应

    index = input_details[0]['index']
    interpreter.set_tensor(index, input_data)
    interpreter.invoke()  # 启动

    output0 = interpreter.get_tensor(output_details[0]['index'])  # bbox
    output1 = interpreter.get_tensor(output_details[1]['index'])  # bbox
    output2 = interpreter.get_tensor(output_details[2]['index'])  # bbox
    output3 = interpreter.get_tensor(output_details[3]['index'])  # 概率
    # 在这里你可以通过print查看4个输出的信息
    # 分别时object_detection的信息：
    # 对于我来讲，人脸检测不涉及类别，所以我只用到
    # output0：位置信息
    # output2：对应的概率

    # 我只要概率最大的人脸，且概率>0.6保持，否则讲概率置为0
    print(output0)
    print(output2)
    # output3 = output3[0][0] if output3[0][0] > 0.01 else 0

    return output0, output2  # 返回概率信息和其位置信息


imgs_path = glob.glob('./JPEGImages/*')

for img_path in imgs_path:
    t1 = time.time()
    img = cv2.imread(img_path)
    sp = img.shape
    bbox, confidence = detection(img)
    # if confidence != 0:
    print('置信度=', confidence, '   bbox=', bbox, end='   ')
    y1 = int(bbox[0][0][0] * sp[0])
    x1 = int(bbox[0][0][1] * sp[1])
    y2 = int(bbox[0][0][2] * sp[0])
    x2 = int(bbox[0][0][3] * sp[1])

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    print('time=', time.time() - t1)
    cv2.namedWindow(str(confidence * 100)[2:6] + '%', 0)
    cv2.imshow(str(confidence * 100)[2:6] + '%', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # else:
    #     print('time=', time.time() - t1)