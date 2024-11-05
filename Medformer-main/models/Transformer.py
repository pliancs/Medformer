import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ConvLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer( #AttentionLayer作为 EncoderLayer的一部分
                        # FullAttention: 这是一个完整的注意力机制实现。
                        FullAttention(  #完整的注意力机制实现。
                            False,  #是否启用因果注意力
                            configs.factor, #缩放因子，用于计算注意力权重
                            attention_dropout=configs.dropout,  # 注意力中的 dropout
                            output_attention=configs.output_attention,  #是否输出注意力矩阵
                        ),
                        configs.d_model,    #设定嵌入的维度，通常对应于模型中每个向量的大小。
                        configs.n_heads,    #指定多头注意力的头数。
                    ),
                    configs.d_model,    #再次指定嵌入的维度，用于层的输入和输出。
                    configs.d_ff,   #指定前馈网络的隐藏层维度，通常比 d_model 大。
                    dropout=configs.dropout,    #前馈网络中使用的 dropout 概率。
                    activation=configs.activation,  #前馈网络中的激活函数类型。
                )
                for l in range(configs.e_layers)  #e_layers 代表模型中的编码器层数（encoder layers）
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model), #通过 for l in range(configs.e_layers)，根据配置的层数 configs.e_layers 来创建多个 EncoderLayer 实例，最终形成一个列表。
        )
        # Decoder   检查任务名称，如果任务为“长期预测”或“短期预测”，则初始化解码器和解码嵌入层 dec_embedding。
        if (    #如果任务是“长期预测”或“短期预测”，则初始化解码器和解码嵌入层，以实现时间序列预测。
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.dec_embedding = DataEmbedding(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.decoder = Decoder( #定义解码器 Decoder，包含 d_layers 个 DecoderLayer，每个 DecoderLayer 包含自注意力、编码器-解码器注意力和前馈网络层。
                [
                    DecoderLayer(
                        AttentionLayer(         #第一层 AttentionLayer: 采用因果注意力机制 FullAttention(True, ...)
                            FullAttention(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        AttentionLayer(     #第二层 AttentionLayer: 不启用因果注意力，用于编码器-解码器之间的注意力
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),    #最后一个线性投影层，将解码器输出维度转换为输出特征数 c_out
            )
        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu   #定义激活函数 GELU（Gaussian Error Linear Unit）。GELU 在神经网络中作为一种平滑的非线性激活函数，通常比 ReLU 更适合复杂任务
            self.dropout = nn.Dropout(configs.dropout)  #定义 Dropout 层来帮助防止过拟合。
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class    #定义全连接层 self.projection 作为最终的输出层，用于将模型的特征向量映射到类别数
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc：编码器的主输入数据（时间序列）。
        # x_mark_enc：编码器时间戳特征（比如日期、小时等辅助特征）。
        # x_dec：解码器的主输入数据。
        # x_mark_dec：解码器时间戳特征。
        # mask：可选的掩码，用于遮蔽部分数据。
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]    #返回的 dec_out 是分类结果，形状为 [B, N]，其中 B 是批大小，N 是类别数。
        return None
