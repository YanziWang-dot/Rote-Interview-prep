# Rote-Interview-prep
# 🚀 Transformer Interview Prep

> 本文系统整理 Transformer 相关 28 个高频面试问题，  

---

## 1. 为什么使用多头注意力机制，而不是一个头？

**回答思路：**

多头注意力的核心目的是让模型能够在**不同子空间中并行建模不同类型的依赖关系**。  
如果只使用单头注意力，模型在一次 attention 计算中只能关注一种关系模式，例如语义相似性或局部依赖。

多头注意力通过将特征维度拆分成多个子空间，每个 head 都拥有独立的 Q、K、V 投影，使不同 head 可以关注不同的信息，比如长距离依赖、局部结构或语法关系。  
最后再将多个 head 的结果拼接融合，在计算量基本不变的情况下显著提升模型表达能力。

---

## 2. 为什么 Q 和 K 使用不同的权重矩阵，而不是用同一个值自身点乘？

**回答思路：**

因为 Query 和 Key 在注意力机制中本身承担的是**不同的语义角色**。  
Query 表示“当前 token 想要查询什么信息”，而 Key 表示“上下文 token 能提供什么信息”。

如果 Q 和 K 使用相同的权重矩阵，attention 会退化为简单的相似度计算，难以建模方向性和非对称关系。  
使用不同的线性映射可以提升模型的灵活性和表达能力，尤其在 Encoder-Decoder Attention 中，Q 和 K 本就来自不同模块，更需要独立建模。

---

## 3. 为什么 attention 选择点乘而不是加法？复杂度和效果有什么区别？

**回答思路：**

从建模能力上看，点乘 attention 和加法 attention 的效果是相近的，但点乘 attention 在计算效率和并行性上具有明显优势。  

点乘 attention 可以直接通过矩阵乘法实现，在 GPU 和 TPU 上高度优化，计算复杂度更低；而加法 attention 需要额外的前馈网络，计算开销更大。  
因此 Transformer 选择点乘 attention，在保证效果的同时大幅提升训练和推理效率。

---

## 4. 为什么在 softmax 之前需要对 attention 进行 scaled（除以 √dₖ）？

**回答思路：**

这是为了防止随着维度增大，点积结果数值过大，从而导致 softmax 梯度消失。  

假设 Q 和 K 的每一维方差为 1，则点积 Q·K 的方差大约为 dₖ。  
当 dₖ 较大时，softmax 输出会非常接近 one-hot，梯度几乎为 0，训练变得不稳定。

因此引入缩放因子：
\[
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

本质上是对 attention score 做方差归一化，稳定训练过程。

---

## 5. 在计算 attention score 时如何对 padding 做 mask？

**回答思路：**

padding 本身不包含语义信息，因此在 attention 计算中需要完全忽略。  
常见做法是在 softmax 之前，将 padding 位置对应的 attention score 加上一个非常小的负值（如 -1e9），使其 softmax 权重趋近于 0。

这样可以确保 padding token 不会对其他 token 的注意力分布产生影响。

---

## 6. 为什么多头注意力中每个 head 需要降维？

**回答思路：**

降维的目的是在**提升表达能力的同时控制计算复杂度**。  

如果每个 head 都使用完整维度，多头 attention 的计算量会随着 head 数线性增长。  
通过将每个 head 的维度设为 `d_model / h`，可以保证整体计算量与单头 attention 基本一致，同时让不同 head 专注于不同子空间的表示学习。

---

## 7. 为什么 embedding 后要乘以 embedding size 的平方根？

**回答思路：**

embedding 初始化时数值范围通常较小，如果直接与位置编码相加，embedding 的信息可能被位置编码淹没。  
通过将 embedding 乘以 √d_model，可以放大其数值尺度，使其与 attention score 和位置编码保持一致，从而提升训练稳定性。

---

## 8. 什么是位置编码？有什么意义和缺点？

**回答思路：**

Transformer 的 self-attention 本身对序列顺序不敏感，因此需要通过位置编码显式引入位置信息。  

位置编码可以是固定的（如 sinusoidal），也可以是可学习的，用于告诉模型 token 在序列中的相对或绝对位置。  
缺点是传统的绝对位置编码对超长序列的泛化能力有限。

---

## 9. 常见位置编码技术及优缺点？

**回答思路：**

- Sinusoidal：无参数、可外推，但只建模绝对位置  
- Learnable PE：灵活，但受训练长度限制  
- Relative PE：建模相对距离，但实现复杂  
- RoPE：支持旋转不变性，适合长序列  
- ALiBi：推理友好，但表达能力有限

---

## 10. Transformer 中的残差结构及其意义？

**回答思路：**

Transformer 使用残差连接将子层输入直接与输出相加，有效缓解梯度消失问题。  
同时残差结构可以让模型更容易学习“恒等映射”，提升深层网络的训练稳定性和收敛速度。

---

## 11. 为什么使用 LayerNorm 而不是 BatchNorm？LayerNorm 在哪里？

**回答思路：**

BatchNorm 依赖 batch size，在 NLP 任务中 batch size 往往较小且序列长度不一致，效果不稳定。  
LayerNorm 对单个样本内部特征做归一化，不依赖 batch，更适合序列建模任务。

LayerNorm 通常放在每个子层的残差连接之后（Post-LN）或之前（Pre-LN）。

---

## 12. BatchNorm 和 LayerNorm 的区别？

**回答思路：**

BatchNorm 在 CV 任务中效果显著，但依赖 batch 统计量，推理和训练行为不一致。  
LayerNorm 对序列任务更稳定，但对尺度变化不敏感。Transformer 中更偏向 LayerNorm。

---

## 13. Transformer 中的前馈神经网络（FFN）是什么？

**回答思路：**

FFN 是对每个位置独立作用的两层全连接网络，用于引入非线性变换。  
常见结构为：
\[
FFN(x)=\text{GELU}(xW_1+b_1)W_2+b_2
\]

FFN 提升了模型的表达能力，但计算量较大。

---

## 14. Encoder 和 Decoder 是如何交互的？

**回答思路：**

Decoder 通过 Encoder-Decoder Attention 与 Encoder 输出交互。  
其中 Query 来自 Decoder，Key 和 Value 来自 Encoder 的最终输出。

---

## 15. Decoder 和 Encoder 的 Self-Attention 有什么区别？

**回答思路：**

Decoder 的 self-attention 使用 causal mask，防止看到未来信息；  
Encoder 的 self-attention 可以看到整个序列。

---

## 16. Transformer 的并行化体现在哪里？

**回答思路：**

Self-attention 不依赖时间步，FFN 对每个 token 独立，因此 Transformer 可以对整个序列并行计算，这是其相比 RNN 的核心优势。

---

## 17. WordPiece 和 BPE 是什么？用过吗？

**回答思路：**

两者都是子词分词算法。  
BPE 基于频率合并子串，WordPiece 基于最大化语言模型概率。  
BERT 系列模型广泛使用 WordPiece。

---

## 18. Transformer 的学习率和 dropout 如何设置？

**回答思路：**

学习率通常采用 warmup + inverse square root decay。  
Dropout 常用于 attention 权重、FFN 和 embedding。  
推理阶段需关闭 dropout。

---

## 19. BERT 为什么不用 attention mask 来做 MLM？

**回答思路：**

attention mask 会完全屏蔽 token，而 MLM 需要保留上下文结构。  
使用 `[MASK]` 可以作为可学习信号，而不是直接切断信息流。

---

## 20. 为什么 self-attention 后要接 FFN？

**回答思路：**

self-attention 负责信息聚合，FFN 负责非线性变换，两者分工明确，类似 CNN 中的卷积加非线性层。

---

## 21. 注意力机制的核心思想是什么？

**回答思路：**

注意力机制的本质是：  
**根据相关性，对信息进行加权选择。**

---

## 22. 自注意力和注意力机制的区别？

**回答思路：**

自注意力的 Q、K、V 来自同一序列；  
普通注意力的 Q 与 K、V 来自不同序列。

---

## 23. Transformer 如何处理长序列？

**回答思路：**

通过稀疏 attention、滑动窗口、线性 attention 或引入外部 memory 来降低 O(n²) 复杂度。

---

## 24. Q、K、V 分别代表什么？

**回答思路：**

Q 表示查询，K 表示索引，V 表示具体内容，是对信息匹配过程的抽象建模。

---

## 25. Transformer 相比 LSTM 在语言模型上的优势？

**回答思路：**

Transformer 支持并行计算，更容易建模长距离依赖，但时间复杂度为 O(n²)。

---

## 26. Transformer 和 BERT 的关系？

**回答思路：**

BERT 是基于 Transformer Encoder 堆叠的预训练模型，使用 MLM 和 NSP 作为训练目标。

---

## 27. Transformer 和 CNN 在图像领域的区别？

**回答思路：**

Transformer 建模全局关系，归纳偏置弱；  
CNN 擅长局部特征，归纳偏置强，对数据需求更低。

---

## 28. Transformer 的模型压缩方法有哪些？

**回答思路：**

常见方法包括剪枝、知识蒸馏、量化和低秩分解，用于降低模型规模和推理成本。

---




