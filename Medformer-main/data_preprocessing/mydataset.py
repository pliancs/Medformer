import os
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle

import re
import cv2
import copy
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import copy

PATH = "C:\\Users/pliancs/Desktop/时间序列/代码仓库/Medformer/Medformer-main/dataset/2. 单脑静息态数据/"
selected_cols = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12, 13, 14, 15, 16, 17, 18, 19]


# 统计 .mat 文件数量
def count_mat_files(path):
    mat_file_count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mat'):
                mat_file_count += 1
    return mat_file_count

list_of_df = []
patient_id=0;
num=count_mat_files(PATH)
print(num)
labels = np.zeros((num, 2))
feature_path = 'Feature'
if not os.path.exists(feature_path):
    os.mkdir(feature_path)
for root, dirs, files in os.walk(PATH):
    for fname in files:
        if fname.endswith('.mat'):  # 只处理 .mat 文件
            data = scipy.io.loadmat(os.path.join(root, fname))  # 加载 .mat 文件
            df_inst = pd.DataFrame(data['pair_data1'])
            # df_inst = df_inst.interpolate(method='linear', limit_direction='forward', axis=0)
            epoch_num = (len(df_inst) - 250) // 125 + 1  # 计算可以切分的轮次数
            print("Epoch number: ", epoch_num)
            # 初始化一个三维数组存放每轮次的分割数据，第三维度大小为20
            temp = np.zeros((epoch_num, 250, 20))
            features = []
            # 使用 dtype=object 创建 labels 数组，以支持字符串

            relative_path = os.path.relpath(root, PATH)
            # 按 250 行为一条数据切分，步长为 125
            for j in range(epoch_num):
                start_idx = j * 125  # 每个子序列的起始索引
                end_idx = start_idx + 250  # 每个子序列的结束索引
                subseq = df_inst.iloc[start_idx:end_idx].values  # 获取每个子序列的值
                temp[j] = subseq  # 存入 temp 数组的第 j 个位置
                features.append(subseq)  # 存入 features 列表
            # 将第一列赋值为字符串
            labels[patient_id][0] = 0 if 'TD' in root else 1
            labels[patient_id][1] = patient_id + 1
            patient_id+=1
            features = np.array(features)  # 将列表转换为数组

            print(f"Filename: {fname}")
            # print(f"Patient ID: {i + 1}")
            print("Raw data:", df_inst.shape)
            print("Temp shape:", temp.shape)  # 查看 temp 数组的形状
            print("Segmented data:", features.shape)  # 查看 features 数组的形状
            np.save(feature_path + "/feature_{:02d}.npy".format(patient_id), features)
            print("Save feature_{:02d}.npy".format(patient_id))
            print()
label_path = 'Label'
if not os.path.exists(label_path):
    os.mkdir(label_path)
np.save(label_path + "/label.npy",labels)
print("Save label")
print(1)
#             label_path = 'Label'
#             if not os.path.exists(label_path):
#                 os.mkdir(label_path)
#             for i in range(len(labels)):
#                 # The first one is AD label (0 for healthy; 1 for AD patient)
#                 # The second one is the subject label (the order of subject, ranging from 1 to 23.
#                 labels[i][1] = i + 1
#                 if i + 1 in AD_positive:
#                     labels[i][0] = 1
#                 else:
#                     labels[i][0] = 0
#             np.save(label_path + "/label.npy", labels)
#             print("Save label")
#
#             df_inst['label'] = 0 if 'TD' in root else 1  # 根据路径名来判断标签
#
#             df_inst['id'] = relative_path + "\\" + fname.split('.')[0]  # 获取文件名（去掉扩展名）
#
#             df_inst = df_inst.reset_index(drop=True)  # 重置索引
#             list_of_df.append(df_inst)  # 添加到列表中
#
#
#
#
#
#
#
#
#             epoch_num=df_inst.shape[0]//125
#             print("Epoch number: ", epoch_num)
#             # Each epoch has shape (1280, 16)
#             temp = np.zeros((epoch_num, 1321, 20))  # 三维数组，epoch_num: 表示数组的第一维大小，1280: 表示数组的第二维大小，16: 表示数组的第三维大小
#             features = []
#             # Store in temp
#             for j in range(epoch_num):  # 一条数据切分为35轮，35轮再切分9条按照250HZ
#                 temp[j] = np.transpose(mat_np[0, 0][2][0][j])
#
#                 # Calculate the number of subsequences that can be extracted
#                 num_subsequences = (temp[j].shape[0] - subseq_length) // stride + 1
#                 # Extract the subsequences
#                 # subquences为一轮切为9条的集合
#                 subsequences = [temp[j][i * stride: i * stride + subseq_length, :] for i in range(num_subsequences)]
#                 feature = np.array(subsequences)
#                 features.append(feature)
#             features = np.array(features).reshape((-1, subseq_length, 16))
#
#             print(f"Filename: {filenames[i]}")
#             print(f"Patient ID: {i + 1}")
#             print("Raw data:", temp.shape)
#             print("Segmented data", features.shape)
#             np.save(feature_path + "/feature_{:02d}.npy".format(i + 1), features)
#             print("Save feature_{:02d}.npy".format(i + 1))
#             print()
#             print(1)
#
#
# def read(path=PATH):
#     list_of_df = []
#     series_name = list(set([x.split('.')[-1] for x in os.listdir(PATH + 'ASD')]))   #series_name存储去重后的文件扩展名
#     for file_type in ['ASD', 'TD']:  #遍历两个字符串 'abnormal' 和 'normal'，代表数据文件的两种类型。不同类型的数据需要分别处理。
#         for fname in list(set([x.split('.')[0] for x in os.listdir(PATH + file_type)])):  #列出当前 file_type 目录下的所有唯一文件名（去掉扩展名），因为有扩展名的原因存在前缀相同的文件7个，for循环选择其中一个
#             dic = {}
#             label = ''  #空字典 dic 用于存储从文件中读取的数据，初始化一个空字符串 label 用于存储当前文件的标签。
#             data = scipy.io.loadmat(PATH + file_type + '/' + fname + '.mat')
#             df_inst = pd.DataFrame(data['pair_data1']) #将字典 dic 转换为 Pandas DataFrame，每个扩展名作为列名，每列包含对应的数据。注意此处将扩展名作为列名
#             # n_paa = 128  # 目标维度
#             # n_timestamps = df_inst.shape[0]
#             # print(n_timestamps)
#             # window_size = n_timestamps // n_paa
#             # paa = PiecewiseAggregateApproximation(window_size=window_size)
#             # paa_results = pd.DataFrame()
#             # for col in df_inst.columns:
#             #     # 转换为二维数组并应用PAA
#             #     data_col = df_inst[col].values.reshape(1, -1)
#             #     X_paa = paa.transform(data_col)[:, :n_paa]
#             #
#             #     # 添加结果到DataFrame
#             #     paa_results[col] = X_paa.flatten()  # 扁平化并添加到结果DataFrame
#             # df_inst=paa_results
#             if(file_type=='ASD'):
#                 df_inst['label'] = 1
#                 df_inst['id'] = fname + str(1)
#             elif(file_type=='TD'):
#                 df_inst['label'] = 0
#                 df_inst['id'] = fname+str(0) #将之前提取的 label 和 fname（文件名）作为新列添加到 DataFrame 中，分别命名为 label 和 id。
#             df_inst = df_inst.reset_index() #重置 DataFrame 的索引，使其从 0 开始，并更新列标签。
#             list_of_df.append(df_inst)  #将当前处理完成的 DataFrame df_inst 添加到列表 list_of_df 中，以便后续进行合并。
#
#     df_agg = pd.concat(list_of_df)  #使用 pd.concat 将 list_of_df 中的所有 DataFrame 合并成一个大的 DataFrame df_agg。到此处为止要处理的数据集彻底生成
# #数据预处理
#     df_agg['target'] = df_agg['label']   #创建一个新列 target，根据 label 列的内容来标记数据：如果标签为 '#FAULT=normal'，则标记为 0（正常），否则标记为 1（异常）。
# #保存id和target的映射关系
#     label_map = dict(
#         zip(df_agg[['id', 'target']].drop_duplicates().id, df_agg[['id', 'target']].drop_duplicates().target))
#
#     return df_agg, label_map
#
# filenames = []
# for filename in os.listdir("AFAVA-AD/"):
#   filenames.append(filename)
# filenames.sort()
# # filenames
# feature_path = 'Feature'
# if not os.path.exists(feature_path):
#     os.mkdir(feature_path)
# print(filenames)
# subseq_length = 256 #定义了子序列的长度，即每个提取的片段包含256个数据点。
# stride = 128  # 定义了每个子序列的步长，设置为128意味着相邻的子序列有50%的重叠。
# for i in range(len(filenames)):
#     # print('Dataset/'+filename)
#     path = "AFAVA-AD/" + filenames[i]
#     mat = sio.loadmat(path)
#     mat_np = mat['data']
#
#     # Get epoch number for each subject
#     epoch_num = len(mat_np[0,0][2][0])   #epoch_num代表可以有多少个批次的数据
#     print("Epoch number: ",epoch_num)
#     # Each epoch has shape (1280, 16)
#     temp = np.zeros((epoch_num, 1280, 16))  #三维数组，epoch_num: 表示数组的第一维大小，1280: 表示数组的第二维大小，16: 表示数组的第三维大小
#     features = []
#     # Store in temp
#     for j in range(epoch_num):  #一条数据切分为35轮，35轮再切分9条按照250HZ
#         temp[j] = np.transpose(mat_np[0,0][2][0][j])
#
#         # Calculate the number of subsequences that can be extracted
#         num_subsequences = (temp[j].shape[0] - subseq_length) // stride + 1
#         # Extract the subsequences
#         #subquences为一轮切为9条的集合
#         subsequences = [temp[j][i * stride : i * stride + subseq_length, :] for i in range(num_subsequences)]
#         feature = np.array(subsequences)
#         features.append(feature)
#     features = np.array(features).reshape((-1, subseq_length, 16))
#
#     print(f"Filename: {filenames[i]}")
#     print(f"Patient ID: {i+1}")
#     print("Raw data:", temp.shape)
#     print("Segmented data", features.shape)
#     np.save(feature_path + "/feature_{:02d}.npy".format(i+1),features)
#     print("Save feature_{:02d}.npy".format(i+1))
#     print()
# AD_positive = [1,3,6,8,9,11,12,13,15,17,19,21]
# labels = np.zeros((23, 2))
# len(labels)
# label_path = 'Label'
# if not os.path.exists(label_path):
#     os.mkdir(label_path)
# for i in range(len(labels)):
#   # The first one is AD label (0 for healthy; 1 for AD patient)
#   # The second one is the subject label (the order of subject, ranging from 1 to 23.
#   labels[i][1] = i + 1
#   if i+1 in AD_positive:
#     labels[i][0] = 1
#   else:
#     labels[i][0] = 0
# np.save(label_path + "/label.npy",labels)
# print("Save label")
#
#
# PATH = "C:\\Users/pliancs/Desktop/给陈校长团队-李开云（教心）-ASD数据/1-movement imitation/2. 单脑数据/2. 单脑静息态数据/Exp1/MF_hand/"
# PATH = "C:\\Users/pliancs/Desktop/给陈校长团队-李开云（教心）-ASD数据/1-movement imitation/2. 单脑数据/2. 单脑静息态数据/"
# PATH = "/groups/g900023/home/u202421100470/pliancs/2. 单脑静息态数据/"
# # PATH = "/groups/g900023/home/u202421100470/pliancs/MF_hand/"
# # selected_cols = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11','12', '13', '14', '15', '16', '17', '18', '19']
# selected_cols = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10,11,12, 13, 14, 15, 16, 17, 18, 19]
# # os.listdir(PATH)
# # print(os.listdir(PATH))
#
#
# # 读取数据的函数
# def read_all(path=PATH):
#     list_of_df = []
#
#     # 使用 os.walk() 递归遍历目录
#     for root, dirs, files in os.walk(path):
#         for fname in files:
#             if fname.endswith('.mat'):  # 只处理 .mat 文件
#                 data = scipy.io.loadmat(os.path.join(root, fname))  # 加载 .mat 文件
#                 df_inst = pd.DataFrame(data['pair_data1'])  # 假设 pair_data1 是数据的关键字
#
#                 # # 在这里应用小波去噪
#                 # for col in df_inst.columns:
#                 #     df_inst[col] = wavelet_denoising(df_inst[col].values)
#                 #     # 检查小波去噪后是否包含 NaN 值
#                 #     if np.isnan(df_inst[col]).any():
#                 #         print(f'Column {col} contains NaN values after wavelet denoising.')
#                 #
#                 #     # 填充处理
#                 #     df_inst[col].fillna(0,
#                 #                         inplace=True)  # 或者选择其他适合的填充方式，例如前向填充：df_inst[col].fillna(method='ffill', inplace=True)
#
#                 #加入PAA
#                 n_paa = 104  # 目标维度
#                 n_timestamps = df_inst.shape[0]
#                 # print(n_timestamps)
#                 window_size = n_timestamps // n_paa
#                 paa = PiecewiseAggregateApproximation(window_size=window_size)
#                 paa_results = pd.DataFrame()
#                 for col in df_inst.columns:
#                     # 转换为二维数组并应用PAA
#                     data_col = df_inst[col].values.reshape(1, -1)
#                     X_paa = paa.transform(data_col)[:, :n_paa]
#
#                     # 添加结果到DataFrame
#                     paa_results[col] = X_paa.flatten()  # 扁平化并添加到结果DataFrame
#                 df_inst = paa_results
#                 df_inst['label'] = 0 if 'TD' in root else 1  # 根据路径名来判断标签
#                 relative_path = os.path.relpath(root, path)
#                 df_inst['id'] = relative_path+"\\"+fname.split('.')[0]  # 获取文件名（去掉扩展名）
#
#                 df_inst = df_inst.reset_index(drop=True)  # 重置索引
#                 list_of_df.append(df_inst)  # 添加到列表中
#
#     df_agg = pd.concat(list_of_df, ignore_index=True)  # 合并所有 DataFrame
#     df_agg['target'] = df_agg['label']  # 创建目标列
#
#     # 保存 ID 和目标的映射关系
#     label_map = dict(
#         zip(df_agg[['id', 'target']].drop_duplicates().id, df_agg[['id', 'target']].drop_duplicates().target)
#     )
#
#     return df_agg, label_map
#
#
# # 调用函数
# df_agg, label_map = read_all()
