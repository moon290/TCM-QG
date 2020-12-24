# TCM-QG
阿里云天池中医大数据竞赛-中医文献问题生成比赛Rank2攻略

# [“万创杯”中医文献问题生成](https://tianchi.aliyun.com/competition/entrance/531826/rankingList)
**参赛选手**：[李婵娟](https://moon290.github.io/)，[罗诚](https://github.com/wulaoshi)

**实验室**：[四川大学，Dilab](https://github.com/dilab-scu)

### 1.赛题背景分析及理解
#### 1.1背景
- 根据上下文 Text 与答案A，输出问题 Q ，一般当做生成式问题来解决。

#### 1.2赛题分析
- 首先，预训练模型强大的语义理解能力使得现在倾向于使用预训练模型来解决问题。

- 其次，数据是中医翻译成白话文数据，我们注意到中医数据与公开的中文预训练模型预训练时使用的语料存在一定的 Gap。
如：中医数据中出现 **《伤寒杂病论》**，我们希望模型在做生成的时候，能让 **伤**、**寒** 这类字或词的 word embedding 更接近，以便能生成更符合中医的句子。但是预训练的语料使用的网上普通语料，这就存在一定差异。为解决以上差异，目前有这几种方式：

    - 融入词汇信息，将n-gram embedding加入每一层，参考 [ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations](https://arxiv.org/abs/1911.00720)。
    - 扩展词表，加权临近词的embedding得到新词表的embedding，参考 [Improving Pre-Trained Multilingual Models with Vocabulary Expansion. CONLL2019](https://arxiv.org/abs/1909.12440)。
    - 直接使用相关语料进行二次预训练。
- 相比第三种方法，前两种方法需要构造垂直领域词汇，需要人工处理，耗时较长，作为低优先备选。
- 基于上述分析，我们的解题思路也明确了：首先选择合适的生成模型，拉近相应预训练模型词向量与中医词向量距离，进行相应的生成。
- 生成模型框架选择：
    - GPT2, 进行 LM 预训练，非常适合做生成任务。
    - Encoder2Decoder，对编码信息捕捉非常全面。

- 根据我们以往经验，对于这类 QA 问题，上下文 Text 信息非常重要，需要模型能较强地捕捉和编码上下文信息，所以我们选择使用 **Seq2Seq** 框架。

### 2.核心思路
#### 2.1模型确定
我们主要使用两类模型分别进行实验：
**1**. 基于医学数据进行预训练的模型 [**wobert**](https://tianchi.aliyun.com/forum/postDetail?postId=130889)。该模型在10+GB的医学文本进一步训练，非常适合用来做医学相关任务。该模型基于 Keras 框架，使用Unilm的方式，通过mask机制用一个wobert完成编码和解码。
**2**. **Seq2Seq** 框架的 **Roberta_wwm_large2transformer**。使用 [**RoBERTa-wwm-ext-large**](
https://github.com/ymcui/Chinese-BERT-wwm) 作为 encoder ，同时使用其 embedding 初始化 decoder 的 embedding 与 ffn_out。
   +  Bert2Bert 模型，如果 encoder, decoder 都用 large 模型，则硬件资源不够训练。如果采用 base 模型，则效果没 Roberta_wwm_large2transformer 好，Roberta_wwm_large2transformer 最为灵活，decoder 可以自行调节。
   + 使用 Roberta_wwm_large 添加 Unilm 的三种注意力矩阵，训练后，效果也没 Roberta_wwm_large2transformer 好。
   
#### 2.2使用其他中文医学QA数据集进行预训练
另外，为了适应QG任务，我们收集了超过 1G 公开医学数据，从中筛选、清洗出了超过 100万条 的符合 Seq2Seq 训练格式数据，进行再次预训练，理论参考：
- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)。
- [TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection](https://arxiv.org/abs/1911.04118)。

#### 预训练方式
**1**. **wobert**：跟用竞赛数据fine-tune时训练方式一致
**2**. **Roberta_wwm_large2transformer**：固定 encoder ，只训练 decoder，能达到 sota 效果。
- 预训练 Bert_base2Bert_base：二次训练，先固定encoder，只预训练decoder；再全部一起，使用不同学习率训练。
- 先预训练 Roberta_wwm_large2transformer 的 decoder，再一起联合预训练，效果不好

#### 2.3数据处理
- 数据统计分析:对训练数据text、question、answer长度进行统计分析，设定最大解码长度为 130。
- 数据清洗:清洗一些有问题的数据，去除一些奇怪特殊符号，诸如"。。。"。
- 截取 Text:观察到问题通常只与答案前面部分text相关，为了减少文本长度，同时避免输入文本中引入一些不相关的信息在构建训练数据时，每条训练数据只保留 "Text" 中从起始位置到 "A" 在 "Text" 中的最后一个结束符号。

#### 2.4训练

- **wobert**：将训练数据构建成 **[cls]Text[sep]Answer[sep]Question** 或者 **[cls]Text1#Answer#Text2[seq]Question** 格式，根据Unilm的形式进行训练。
- **Roberta_wwm_large2transformer**：将训练数据构建成 **[Text[sep]Answer[sep], [cls]]Question[sep]]** 对数据，进行 Seq2Seq 训练。

#### 2.5实验效果
- 单模型解码：经过二次预训练的模型，在微调后，**Roberta_wwm_large2transformer** 单模型在复赛 Test2 上的 
ROUGE-L 为 0.6205 左右；**wobert**  单模型在复赛 Test2 上的 ROUGE-L 为 0.6121  左右；
- 模型融合：我们一共融合了7个模型结果，分别是 4个基于**wobert** 的不同模型与 三个**Roberta_wwm_large2transformer**不同模型，线上达到0.6332，融合效果提升明显。

#### 2.6一些尝试
**1**. Roberta_wwm_large 添加倒三角 mask attention，使用 Roberta 模仿 Unilm，集成 Seq2Seq，dev: 0.5771，无用。
**2**. 使用 Roberta_wwm_base2Roberta_wwm_base，即编码器、解码器都使用 Roberta_wwm_large_base，dev: 0.52，无用。
**3**. 在 embedding 层添加 answer_embedding，即除了 [word_embedding，position_embedding，segment_embedding] 外，再添加了一个 answer_embedding， answer 处为1，其余为0。并把数据处理成 [cls]Text[sep]，其中 Text 中处理成 xxxx[begin]answer[end]xxx，即在 Text 中在 A 的前后处加上特殊符号。效果没 [Text[sep]Answer[sep] 数据处理效果好，无用。
**4**. Roberta_wwm_large2transformer，在下一条的基础上，只预训练 decoder 部分，test2：0.6205，有用。
**5**. Seq2Seq 架构，用 encoder 的 embedding 初始化 decoder：
- encoder embedding 与 decoder embedding 共享参数，有提升；
- encoder embedding 与 decoder embedding 不共享参数，比上一条提升更多；
- encoder embedding 与 decoder embedding 不共享参数，并再用 encoder embedding 初始化 decoder ffn_out 的参数，比上一条提升更多；
- encoder embedding 与 decoder embedding 不共享参数，并再用 decoder embedding 共享 decoder ffn_out 的参数，没上一条提升多；

**6**. Roberta_wwm_large2transformer 参数全部进行预训练：dev: 0.628，test2：0.5860，验证集提升，但是 test2 却下降，无用；
**7**. ensemble：选取几个模型进行 rerank 融合，即几个模型的生成结果两两计算ROUGE-L分数并求和，取其中 ROUGE-L 最高的结果作为最终结果，有用。

### 3.比赛总结和感想
- 用开源的中文医学QA数据进行预学习，在线下提升较大，但线上提升较小，可能是没有根据任务进行精细化设计。
- 使用预训练模型，虽然效果好，但耗时较长，需要较多算力。因为时间和硬件原因，还有不少想法未测试验证或尝试。
- 比赛过程中经常出现线下分数上升，但是线上分数下降的情况，说明自行划分的数据集分布不太一致，需要增强模型鲁棒性，本打算尝试对抗训练，如 FGM、PGD，但是提交次数和硬件资源告急，未尝试。
- 队友合作互助很重要，最好相互代码检查。面对需要进行长时间训练的模型代码，最好能相互**审核**一下代码，避免训练完才发现有BUGGGGG，浪费大量时间。
- 前期讨论规划很重要。打比赛需要有规划，最好列一个时间线，分成不同阶段，有选择地尝试，合理估计每个想法可能的提升，有所取舍，并规定每个阶段的 **deadline**。





