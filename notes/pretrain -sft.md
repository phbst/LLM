## 一点个人理解

这周手写了一便llama预训练、T5的sft。记录

我把他们都理解为：使用优化器、根据训练数据推理的得到loss，反向传播更新模型参数。

总结为四主体：
- 模型
- 数据
- 优化器
- 损失函数

## 模型层
从transformer到deepseek这一过程。都围绕提高性能、降低成本开展。

分词采用BBPE允许更多陌生词引入

为了增加上下文窗口，改进位置编码到Rope、ALiBi，增加外推性

为了提高性能，引入了attention机制，、使得模型可以并行计算，提高计算效率。

同时从MHA -> MQA -> GQA -> MLA。不断在减少计算成本而不影响性能。

从batchnorm/layernorm -> RMSnorm

在前馈层引入MOE专家网络，模型稀疏化，减少计算量，提高性能。

采样从Next token prediction 走向 MTP。

确立了 模型Pretrain -> SFT -> RL的模式

## 数据层

数据质量决定模型上限：

质量过滤：可以使用分类器过滤，或者基于规则、关键词集合、困惑度等。同时考虑效率

敏感内容：分类器与启发式的关键词识别、敏感内容邮箱、电话等替换/删除

去重：分不同级别、文档、句子间。考虑字符之间的相似/精确匹配

## 优化器层

目前研究不多，但是主流是AdamW

## 损失函数层

目前研究不多，根据不同任务来决定，我用交叉商。