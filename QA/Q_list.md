说一下 ChatGPT的优缺点

2请简述下Transformer基本流程

3为什么基于Transformer的架构需要多头注意力机制？

4编码器，解码器，编解码LLM模型之间的区别是什么？

5你能解释在语言模型中强化学习的概念吗？它如何应用于ChatGPT？

6在GPT模型中，什么是温度系数？

7什么是旋转位置编码（ROPE）？

8为什么现在的大模型大多是decoder-only的架构？

9ChatGPT的训练步骤有哪些？

10为什么transformers需要位置编码？

11为什么对于ChatGPT而言，提示工程很重要？

12如何缓解 LLMs 复读机问题？

---

解释一下langchain Agent的概念

2langchain的6大核心组件是什么，每个组件有什么作用？

3langchain有哪些优点和明显的缺点？

4langchain有哪些替代方案？

5什么是检索增强生成（RAG）？

6在做知识增强检索时，文本切分有哪些方法？

7目前主流的中文向量模型有哪些？

8相比模型直接生成，RAG的优势是什么？

9SELF-RAG是什么，SELF-RAG如何提升大型语言模型的质量和准确性？

10RAG和微调的区别是什么？

11什么是 Graph RAG？

---

1Prompt design，Prompt tuning 还是 finetuning的区别是什么？

2参数高效的fine-tuning(PEFT)是什么？

3介绍一下prompt-tuning技术

4什么是Prefix tuning？

5介绍一下LORA微调

6相比LORA，AdaLORA的改进点是什么？

7QLORA模型有什么创新点？

8稀疏微调是怎么工作的，有哪几个步骤？

9监督微调SFT后LLM表现下降的原因

10什么是P-Tuning？

11多轮对话任务如何微调模型？

---

1请简述下PPO算法。

2介绍一下基于人类反馈的强化学习流程

3奖励模型的数据收集要满足什么要求？

4奖励模型是如何训练的，它的损失函数是什么？

5目前RLHF 方法有没有什么缺陷？如何改进

6介绍一下LLM的直接偏好优化（DPO）

7LLM训练中，近端策略优化包含哪几个模型？

8什么是RLAIF？

9与有监督学习相比，强化学习能够给大语言模型带什么哪些好处？

10介绍一下RLHF中PPO微调过程

11你了解DeepMind提出的ReST对齐算法吗？

---

1说说你知道的大模型训练or推理的常用优化手段

2一般会对哪些大模型里面的算子做算子融合，说说你知道的

3什么是KV Cache技术，它具体是如何实现的？

4Paged Attention的原理是什么，它解决了大模型推理中的什么问题？

5DeepSpeed 推理对算子融合做了哪些优化？

6FlashAttention的空间复杂度和对HBM的访问次数是多少？

7FlashDecoding在FlashAttention2上做了哪些改进？

8FlashDecoding++做了什么优化？

9什么是子图融合优化技术，为什么他可以提升推理速度？

10MHA，GQA，MQA推理优化技术的区别是什么？

11Paged Attention是如何有效管理具有分页的KV缓存的？

12介绍一下动态批处理技术

13请问什么是猜测推理技术？请举例说明

14什么是continuous batching技术，为什么他的效率比动态batching效率高？

15优化CUDA程序的访存效率，你可以想到哪些？

16优化CUDA程序的计算效率，你又可以想到哪些？