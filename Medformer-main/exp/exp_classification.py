from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag="TEST")    #下面的_get_data函数
        self.args.seq_len = test_data.max_seq_len  #seq_len 设置为测试数据的最大序列长度。
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  #enc_in 为输入特征的维度。（样本数，时间步数，特征数）
        self.args.num_class = len(np.unique(test_data.y))   #num_class 表示分类任务的类别数量。
        # model init    从 model_dict 中获取模型类并初始化，转换为 float 类型。
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:   #使用多 GPU 时，模型通过 DataParallel 并行化处理。
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)  # 设置随机种子
        data_set, data_loader = data_provider(self.args, flag)  ## 获取数据集和加载器    data_factory里面的data_provider类
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)   # Adam优化器
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()   ## 交叉熵损失函数
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0  #用于记录每个epoch中的迭代次数。iter_count 的作用是用于计算训练过程中每 100 次迭代的平均速度，并估计剩余训练时间。
            train_loss = [] #存储当前epoch内的每次迭代的损失。

            self.model.train()  #将模型设置为训练模式，使其能够执行梯度更新等操作。
            epoch_time = time.time()    #记录当前epoch开始的时间，用于计算时间消耗。

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                # 遍历数据加载器中的训练数据，train_loader提供批量的输入数据batch_x、标签label和padding_mask。
                iter_count += 1
                model_optim.zero_grad() #model_optim.zero_grad()：清除优化器的梯度，以防止累积。
                # 将数据和标签移动到指定设备上（如GPU），并转换为浮点型。
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None) #前向传播计算模型输出outputs。
                loss = criterion(outputs, label.long()) #使用损失函数criterion计算损失loss，并存入train_loss列表。
                train_loss.append(loss.item())
                #迭代次数为100的整数
                if (i + 1) % 100 == 0:
                    # 打印日志
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    # 打印迭代速度和估计剩余时间
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            # 更新StochasticWeightAveraging（SWA）模型的参数，以稳定训练结果（如果启用了SWA）。
            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))    #Epoch: <n> cost time: 当前epoch的训练耗时。
            train_loss = np.average(train_loss)
            train_loss, train_metrics_dict = self.vali(train_data, train_loader, criterion)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)
            # 打印训练、验证、测试集的损失和各项指标
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"   #Train Loss: 当前epoch在训练集上的平均损失值。
                f"Training results --- Loss: {train_loss:.5f}, "
                f"Accuracy: {train_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {train_metrics_dict['Precision']:.5f}, "
                f"Recall: {train_metrics_dict['Recall']:.5f}, "
                f"F1: {train_metrics_dict['F1']:.5f}, "
                f"AUROC: {train_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {train_metrics_dict['AUPRC']:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "   #验证结果：在验证集上的各项指标，Loss：损失值。
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, " #AUROC：ROC曲线下的面积。
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            # 使用early_stopping进行早停检查，若验证集F1分数不再提高，则提前终止训练。
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""
        # 如果启用了SWA，则加载SWA模型的最佳权重；否则加载普通模型的最佳权重。
        best_model_path = path + "checkpoint.pth"
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return test_metrics_dict['Accuracy']
