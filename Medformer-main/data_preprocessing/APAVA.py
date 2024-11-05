import os
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
filenames = []
for filename in os.listdir("AFAVA-AD/"):
  filenames.append(filename)
filenames.sort()
# filenames
feature_path = 'Feature'
if not os.path.exists(feature_path):
    os.mkdir(feature_path)
print(filenames)
subseq_length = 256 #定义了子序列的长度，即每个提取的片段包含256个数据点。
stride = 128  # 定义了每个子序列的步长，设置为128意味着相邻的子序列有50%的重叠。

for i in range(len(filenames)):
    # print('Dataset/'+filename)
    path = "AFAVA-AD/" + filenames[i]
    mat = sio.loadmat(path)
    mat_np = mat['data']

    # Get epoch number for each subject
    epoch_num = len(mat_np[0,0][2][0])   #epoch_num代表可以有多少个批次的数据
    print("Epoch number: ",epoch_num)
    # Each epoch has shape (1280, 16)
    temp = np.zeros((epoch_num, 1280, 16))  #三维数组，epoch_num: 表示数组的第一维大小，1280: 表示数组的第二维大小，16: 表示数组的第三维大小
    features = []
    # Store in temp
    for j in range(epoch_num):  #一条数据切分为35轮，35轮再切分9条按照250HZ
        temp[j] = np.transpose(mat_np[0,0][2][0][j])

        # Calculate the number of subsequences that can be extracted
        num_subsequences = (temp[j].shape[0] - subseq_length) // stride + 1
        # Extract the subsequences
        #subquences为一轮切为9条的集合
        subsequences = [temp[j][i * stride : i * stride + subseq_length, :] for i in range(num_subsequences)]
        feature = np.array(subsequences)
        features.append(feature)
    features = np.array(features).reshape((-1, subseq_length, 16))

    print(f"Filename: {filenames[i]}")
    print(f"Patient ID: {i+1}")
    print("Raw data:", temp.shape)
    print("Segmented data", features.shape)
    np.save(feature_path + "/feature_{:02d}.npy".format(i+1),features)
    print("Save feature_{:02d}.npy".format(i+1))
    print()
AD_positive = [1,3,6,8,9,11,12,13,15,17,19,21]
labels = np.zeros((23, 2))
len(labels)
label_path = 'Label'
if not os.path.exists(label_path):
    os.mkdir(label_path)
for i in range(len(labels)):
  # The first one is AD label (0 for healthy; 1 for AD patient)
  # The second one is the subject label (the order of subject, ranging from 1 to 23.
  labels[i][1] = i + 1
  if i+1 in AD_positive:
    labels[i][0] = 1
  else:
    labels[i][0] = 0
np.save(label_path + "/label.npy",labels)
print("Save label")
