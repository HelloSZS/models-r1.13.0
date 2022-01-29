# xml2csv.py

# import os
# import glob
# import pandas as pd
# import xml.etree.ElementTree as ET
#
# # os.chdir('C:\Users\29533\Desktop\DefectDetection\nanodet-main\dataset\Annotations-obj')
# path = r'C:\Users\29533\Desktop\DefectDetection\nanodet-main\dataset\Annotations-obj'
#
# def xml_to_csv(path):
#     xml_list = []
#     for xml_file in glob.glob(path + '/*.xml'):
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         for member in root.findall('object'):
#             value = (root.find('filename').text,
#                      int(root.find('size')[0].text),
#                      int(root.find('size')[1].text),
#                      member[0].text,
#                      int(member[4][0].text),
#                      int(member[4][1].text),
#                      int(member[4][2].text),
#                      int(member[4][3].text)
#                      )
#             xml_list.append(value)
#     column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
#     xml_df = pd.DataFrame(xml_list, columns=column_name)
#     return xml_df
#
#
# def main():
#     image_path = path
#     xml_df = xml_to_csv(image_path)
#     xml_df.to_csv('test.csv', index=None) #需要更改
#     print('Successfully converted xml to csv.')
#
#
# main()


import os
import random
import time
import shutil

xmlfilepath=r'./Annotations'
saveBasePath=r"./Annotations"

# train+val占总数据集的比例:
trainval_percent=0.8

# train占train+val的比例:
train_percent=0.8

total_xml = os.listdir(xmlfilepath)

# xml的数量
num = len(total_xml)

# list:[0, 1, ..., num-1]
list = range(num)

# train+val的数据数量: tv。tv占总数据集的80%，trainval_percent = 0.8
# val的数量
tv = int(num*trainval_percent)

# train的的数据数量:tr。tr占train+val的80%，train_percent = 0.8
# train的数量
tr = int(tv*train_percent)


# 取"数据集名字"列表中的数据名字，用index来取
# "数据集名字"列表：total_xml = ['IMG_1024.xml', 'IMG_1026.xml', 'WIN_20211127_17_20_56_Pro.xml', 'WIN_20211127_17_21_04_Pro.xml', ... ,'WIN_20211127_17_31_28_Pro.xml']
# total_xml[0] 根据index: 0, 就会取出第一个数据名字字符串：'IMG_1024.xml'

# 这里trainval和train为index列表(索引列表)
# 从list里面取train+val个元素
# 例子：
# list = [0,1,2,3,4,5]
# tv = 2
# trainval = random.sample(list,tv)
# trianval 会等于 [element1, element2]，element1和element2取自[0,1,2,3,4,5]，两者不重复

# 在下述代码中，假如trainval取得[0,1,2,3], train取得[1, 2], 那么剩下的[0, 3]就会作为val的index_list
trainval = random.sample(list,tv)
train = random.sample(trainval,tr)

print("train and val size",tv)
print("train size",tr)

start = time.time()

test_num=0
val_num=0
train_num=0

for i in list:
    name=total_xml[i]
    if i in trainval:  #train and val set
        if i in train:
            directory="train"
            train_num += 1
            xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))
            if(not os.path.exists(xml_path)):
                os.mkdir(xml_path)
            filePath=os.path.join(xmlfilepath,name)
            newfile=os.path.join(saveBasePath,os.path.join(directory,name))
            shutil.copyfile(filePath, newfile)
        else:
            directory="validation"
            xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))
            if(not os.path.exists(xml_path)):
                os.mkdir(xml_path)
            val_num += 1
            filePath=os.path.join(xmlfilepath,name)
            newfile=os.path.join(saveBasePath,os.path.join(directory,name))
            shutil.copyfile(filePath, newfile)

    else:
        directory="test"
        xml_path = os.path.join(os.getcwd(), 'Annotations/{}'.format(directory))
        if(not os.path.exists(xml_path)):
                os.mkdir(xml_path)
        test_num += 1
        filePath=os.path.join(xmlfilepath,name)
        newfile=os.path.join(saveBasePath,os.path.join(directory,name))
        shutil.copyfile(filePath, newfile)

end = time.time()
seconds=end-start
print("train total : "+str(train_num))
print("validation total : "+str(val_num))
print("test total : "+str(test_num))
total_num=train_num+val_num+test_num
print("total number : "+str(total_num))
print( "Time taken : {0} seconds".format(seconds))