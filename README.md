# Rote-Interview-prep
Transformer
1.为什么使用多头注意力机制，而不是一个头
2.为什么Q和K使用不同的权重矩阵生成，而不是用同一个值自身的点乘？
3.计算attention的时候为什么选择点乘而不是加法？两者在计算复杂度和效果上有什么区别？
4.为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解
5.在计算attention score的时候如何对padding做mask操作？
6.为什么在进行多头注意力的时候需要对每个head进行降维？
7.为何在获取词输入向量之后需要对矩阵乘以embedding size的开方?意义是什么？
8.简单介绍一下位置编码？有什么意义和缺点？
9.你还了解哪些关于位置编码的技术？各自的优缺点是什么？
10.简单讲一下Transformer中的残差结构以及意义？
11.为什么transformer中的残差结构是LayerNorm而不是BatchNorm？LayerNorm在Transformer的位置是哪里？
12.讲一下BatchNorm，以及它的优缺点。 拓展LayerNorm。。
13.描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？
14.Encoder端和Decoder端是如何进行交互的？
15.Decoder阶段的多头自注意力和Encoder的多头自注意力有什么区别？
16.Transformer的并行化体现在哪个地方？
17.描述一下wordpiece model和byte pair encoding，有实际应用过吗？
18.Transformer训练的时候学习率是如何设定的？dropout是如何设定的？位置在哪里？dropout在测试的时候需要有什么需要注意的吗？
19.Bert的mask为什么不学习transformer在attention处进行屏蔽score的技巧？
20.为什么Transformer的self attention后要使用一个ffn？
21.注意力机制的核心思想是什么？
22.自注意力机制和注意力机制的区别是什么？
23.Transformer如何处理长序列数据？
24.Transformer中的Q、K、V矩阵分别代表什么？
25.Transformer和LSTM相比，在语言模型任务上有什么优势？
26.Transformer和Bert的关系是什么？
27.在图像识别领域，Transformer和CNN相比有哪些不同？
28.Transformer中如何进行模型压缩



