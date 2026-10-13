# 文献与作者代码核查目录

核查截点：**2026-10-13**。来源包括 [飞书 revision 12 原稿](sources/feishu-kda-rev12.md)及本会话后续一手论文/作者代码阅读。本文记录论证边界，不是跨论文排行榜；本项目没有新增实验结果。

证据标签约定：

- **论文已读**：本会话阅读了相关论文机制/实验/限制；不表示所有结论被独立复现。
- **代码已查**：阅读了作者公开实现相关路径；不表示安装、测试、训练或部署成功。
- **原稿入口**：保留原稿引用，具体版本/全文/代码仍需在复现前补核。
- **未定位作者代码**：截至讨论时没有找到可靠公开入口；不是断言任何代码都不存在。

模型规模、训练配方、状态精度、窗口、混合层与测试集不同，不能直接拿各论文分数判断 A/B/C 胜负。未来实验需固定论文版本、代码提交、配置和数据版本；代码仓库 `main` 链接仅作入口。

## 1. 对项目决策最直接的核查

### MoM：B 必须面对的直接近邻

- 论文：[MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/html/2502.13685)，讨论阅读版本 v4。
- 作者代码：[OpenSparseLLMs/MoM](https://github.com/OpenSparseLLMs/MoM)，相关 [layer](https://github.com/OpenSparseLLMs/MoM/blob/main/mom/layers/mom.py) 和 [模型/负载均衡](https://github.com/OpenSparseLLMs/MoM/blob/main/mom/models/mom/modeling_mom.py)。另有 [Linear-MoE](https://github.com/OpenSparseLLMs/Linear-MoE) 入口。
- 状态：论文已读、相关作者代码已查；未运行。FLA 官方仓库已集成 MoM 实现入口。
- 关键机制：多独立递归记忆、输入 top-k 路由、非激活记忆保持、共享记忆与负载均衡。多 bank 与不衰减本身不是 B 的创新。
- 预算边界：§4.2.3 与 Appendix G.2 有“Equal Memory Capacity”对照，但正文以扩大 value 维度匹配 **all activated memories**；主配置是 4 个 routed memory、top-2 及 shared memory。不能当成整个持久 pool 加全部管理状态的同真实字节验证，也不能因此把记忆组织研究说成空白。

### SDM：稀疏访问大状态，不是同状态容量证据

- 论文：[Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity](https://arxiv.org/html/2607.07386)。
- 作者代码：[facebookresearch/sparse-delta-memory](https://github.com/facebookresearch/sparse-delta-memory)，相关 [layer](https://github.com/facebookresearch/sparse-delta-memory/blob/main/lingua/sparse_delta_memory/layer.py)、[decode state](https://github.com/facebookresearch/sparse-delta-memory/blob/main/lingua/sparse_delta_memory/cache.py)。
- 状态：论文已读、作者实现/缓存路径已查；公开有模型定义及 Triton/CUDA 内核，未运行。本会话核对仓库提交为 `183e7df809131b80ad4393741029d0f20fc3640b`。
- 关键机制：Product-Key 寻址，稀疏读写及 delta 更新；只对被写槽施加门控，未选槽保持。其稀疏寻址与抗竞争组织已经存在。
- 实验边界：主要对照为相同 FLOPs 与非记忆参数，训练包含局部 SWA 与全局递归层。论文 Table 1 的 8.14B 档，GDN 状态约 2.2M、SDM 约 7.963B 标量；大状态训练和长上下文收益不能证明 B 在同字节下成功。论文明确承认状态显存可接近模型参数规模及内核仍需改进。

### GSA/GSA2：slot 两侧关联与纠错

- GSA 论文：[Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146)，[NeurIPS 2024 正文](https://papers.neurips.cc/paper_files/paper/2024/file/d3f39e51f5f634fb16cc3e658f8512b9-Paper-Conference.pdf)。官方实现入口：[FLA GSA layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gsa.py)。
- GSA2 论文：[Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention](https://arxiv.org/html/2610.02816)，讨论阅读 v1（2026-10-02）。
- 状态：GSA2 论文已读；全文与作者主页检索未定位可靠作者代码入口。已有 GSA 实现不等于 GSA2 已公开复现。
- GSA2 机制：\(S_1\in\mathbb R^{M\times D_k}\)、\(S_2\in\mathbb R^{D_v\times M}\) 共享 latent-slot 表示，key 侧 Oja-2 与 value 侧 GDN2 纠错，独立擦写门；读取 \(S_2\operatorname{SiLU}(S_1q)\)。这是两级 dense slot 记忆，不应写成与 B 相同的 top-k bank 路由。
- 实验边界：报告约 1.3B/100B tokens 的递归与 SWA 混合实验；部分基线分数沿用 GDN2。参数匹配/吞吐比较不能自动视为整套状态字节匹配；2K 真实检索与2K SWA窗口重合时，局部精确路径能直接访问大量证据。

### NHA：内容读取融合已存在，公开解码路径需单列

- 论文：[Native Hybrid Attention for Efficient Sequence Modeling](https://arxiv.org/html/2510.07019)，讨论阅读 v3。
- 作者代码：[JusenD/NHA](https://github.com/JusenD/NHA)，[截至截点可核版本的 layer](https://github.com/JusenD/NHA/blob/bf85d2db6aee27a7541c9009acbffacc99d40b91/nha_fla/layers/nha.py)。
- 状态：论文已读、相关作者代码已查；未运行。标准 layer 公开版本在有历史 cache 的单 token decode 时抛 `NotImplementedError`，对应 2026-09-14 提交 `bf85d2db6aee27a7541c9009acbffacc99d40b91`。这限制该路径的公开工程完成度，不否定论文结果，也不推广为所有变体不可推理。
- 关键机制：长期递归 key/value slots 与近期精确 tokens 拼接，用统一 softmax 内容读取；只有移出窗口的 token 更新长期记忆。
- 实验边界：论文有小模型预训练及 Llama/Qwen 大模型混合适配。全注意力层、窗口 KV 和两个 slot 状态都应计费；适配质量和显存/延迟曲线不是与新 B 同字节的比较。README 的加速声明与实际可运行路径须分开核查。

### PKDA、OWM、GPM：A 的公式与保护思路强重叠

| 来源 | 一手论文 | 作者代码/核查状态 | 对 A 的约束 |
| --- | --- | --- | --- |
| PKDA / Preconditioned DeltaNet | [正文](https://arxiv.org/html/2604.21100) | [ntumm120/preconditioned-deltanet](https://github.com/ntumm120/preconditioned-deltanet)：本会话已查 `precond_kda` 实现及测试入口，未运行 | 历史 key Gram、预条件与二次保护已有联系；低秩替代对角或逆矩阵公式不是充分创新 |
| OWM | [Continual Learning of Context-dependent Processing in Neural Networks](https://arxiv.org/pdf/1810.01256) | 论文已读；本文不补列未经本会话核查的作者代码 | 限制更新沿已保护输入子空间变化的思想已存在；其权重持续学习证据不是 KDA 请求状态证据 |
| GPM | [Gradient Projection Memory for Continual Learning](https://arxiv.org/abs/2103.09762) | 论文已读；作者代码入口待补核 | 存重要表示子空间并投影梯度已有先例；应用对象不同不能让“正交保护”重新成为空白 |
| Gated KalmaNet | [正文](https://arxiv.org/html/2511.21016) | [awslabs/hybrid-model-factory](https://github.com/awslabs/hybrid-model-factory)：本会话相关代码已查，未运行 | 在线二阶回归、历史统计与遗忘是近邻 |
| Kalman Delta Networks | [正文](https://arxiv.org/html/2609.07816) | [ngocbh/kalman-delta-networks](https://github.com/ngocbh/kalman-delta-networks)：本会话相关代码已查，未运行 | 协方差/不确定性驱动写入已有研究；有限地址采样应单独说明差异与代价 |

剩余研究空间必须落在有限因果保护对象的选择、合法释放、同预算效果及高效实现上；不能用跨论文平均分证明 A 新颖。

### HLA、HLA-WM 与删除/版本近邻：C 的边界

| 来源 | 一手论文 | 作者代码/核查状态 | 对 C 的约束 |
| --- | --- | --- | --- |
| HLA | [v1 正文](https://arxiv.org/html/2610.05842v1) | [Caesarhhh/HLA_](https://github.com/Caesarhhh/HLA_)：本会话核对 Oct-8 提交 `5879c5`；未运行 | 查询条件化历史仿射块重组已有。核对的 LM 主路径保留全历史缓存，未接入论文 \(\theta=0.1\) 硬截断；独立 tile-top-k 内核存在不等于主路径已交付稀疏读取 |
| HLA-WM | [v1 正文](https://arxiv.org/html/2610.05739v1) | 论文已读；作者代码入口本文未补核 | 视频世界模型中的 sink+recent+geometry top-k；不能直接充当语言事实修订证据 |
| Exact Record Omission | [v3 正文](https://arxiv.org/html/2609.06872v3) | 论文已读；代码入口本文未补核 | 区分不再输出与 never-stored 状态；冻结输入轨迹与神经上下文的条件不同，重组旧摘要不是整网精确删除 |
| VersionRAG | [Version-Aware Retrieval-Augmented Generation for Evolving Documents](https://arxiv.org/html/2510.08109v1) | 论文已读；本文未验证部署实现 | 文档版本图、变化与时序有效性检索已有；版本敏感 QA 不是新的业务问题，也不等于固定字节递归状态问题 |
| SmartVector | [Self-Aware Vector Embeddings for Retrieval-Augmented Generation](https://arxiv.org/html/2604.20598v1) | 论文已读；小型合成实验，不能标为成熟系统 | 时间、置信度与关系驱动知识维护为 C 的外部近邻；不能拿其存在证明所有真实修订已解决 |
| Mem0 | [Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/html/2504.19413) | 本会话论文已读；本文不新增代码/部署核查 | 外部记忆的抽取、整合和语义更新已有；仍依赖关联匹配与更新判断，不能当成任意修订正确性的保证，也不是同总字节 KDA 对照 |
| Zep | [A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/html/2501.13956) | 本会话论文已读；本文不新增代码/部署核查 | 时间图与失效/更新关系已有；自然语言事实绑定和时序判断仍有错误边界，外部持久图及检索成本不能放在预算外 |

C 的修订后继闭包不是充分正确性条件；每块 \(A_j+B_j\) 存储昂贵，依赖识别、缺失前驱和原轨迹隐表示都会影响重组。成熟结构化版本表/日志应作为独立业务基线，但不能把上表小型研究一律称为生产成熟。

## 2. 原稿基础与其他强近邻入口

以下保留 revision 12 的研究脉络，并补入后续讨论已读的 BASED。没有明确标记“论文已读”或单列代码核查的行，属于**原稿入口**；不表示本次整理已经读完全文或验证公开实现。FLA 是优先复现入口：[官方仓库](https://github.com/fla-org/flash-linear-attention)，相关实现需按实验时的确切提交固定。

| 工作 | 一手论文 | 本项目用途/需要避免的误读 |
| --- | --- | --- |
| Linear Attention | [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | 固定状态与递归读取基础 |
| RetNet | [Retentive Network](https://arxiv.org/abs/2307.08621) | 衰减递归基线 |
| GLA | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | 数据依赖门与高效块训练基线 |
| DeltaNet chunk training | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | delta 递推及 chunk 实现来源 |
| GDN | [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | 纠错+遗忘强基线；不能将其数值直接移植给 KDA |
| BASED | [Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff](https://arxiv.org/html/2402.18668) | 本会话论文已读；有限状态 MQAR 下界与同 state 的召回曲线是问题定位依据；结论依赖具体任务/状态设定，不能推广成自然语言任意查询的统一容量定理 |
| Kimi Linear / KDA | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | 核心 KDA 与混合架构；[Kimi 作者仓库](https://github.com/MoonshotAI/Kimi-Linear)与[FLA KDA layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py) 为实现入口，需按实际复现路径固定版本 |
| Kimi K3 | [Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) | 成熟混合模型参照；整体长上下文能力不能全归给纯 KDA |
| FG2 | [论文](https://arxiv.org/abs/2604.19021) | 更细门控已有近邻；单纯加门不够 |
| GDN2 | [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | 擦除/写入解耦强基线 |
| EDA | [Erase-then-Delta Attention](https://arxiv.org/abs/2606.26560) | 擦除/写入地址解耦；覆盖与保持需分别测量 |
| QED | [The Query Knows What to Forget](https://arxiv.org/abs/2608.13668) | 当前 query 感知附加擦除；与 A 多历史地址保护目标不同，但宽泛正交/query主张已有 |
| Lattice | [论文](https://arxiv.org/abs/2504.05646) | 正交更新/降低冗余已有邻近研究 |
| CKDA | [论文](https://arxiv.org/abs/2609.24797) | 状态控制与表达能力；跟踪改善不等于任意精确历史访问 |
| HOLA | [论文](https://arxiv.org/abs/2607.02303) | 有限精确 KV 与 delta 状态；选择策略和额外缓存容量均须计入 |
| How Linear Attention Remembers | [论文](https://arxiv.org/abs/2609.33093) | 分离距离与写入竞争的机制诊断；GLA/GDN 因果证据不能直接当纯 KDA 新实验 |

## 3. 原稿数值参照：仅用于保留讨论依据

本表转录 revision 12 的条件性参照；不构成本项目复现结果，本次归档没有逐表新增独立数值审计。引用原文是下一次复现前的复核入口。

| 原论文条件（原稿记录） | 原稿记录的量级 | 不能推出 |
| --- | --- | --- |
| [QED](https://arxiv.org/abs/2608.13668)：340M，15B tokens，2K训练，两个种子 | 8K S-NIAH-1，GDN2→QED 为 64.4→91.6、50.6→84.2；验证loss无明显改善 | 高密度多事实、合法覆盖、总体语言能力均已解决 |
| [HOLA](https://arxiv.org/abs/2607.02303)：340M，15B tokens，2K窗口 | 32K单针：GDN14%、同容量近期缓存24%、HOLA58% | 主规模单种子的单针改善等于全任务胜出；有限KV能保存所有重要历史 |
| [SDM](https://arxiv.org/abs/2607.07386)：8.14B及长上下文训练 | 跨长度RULER均分GDN34.2、SDM50.2；状态约2.2M与7.963B标量 | 同持久字节的B增益成立 |
| [HLA](https://arxiv.org/abs/2610.05842)：1.3B，100B tokens，4K训练 | 32K单针11.2→67.4；全任务平均3.65→7.87；部分追踪/词抽取下降 | 修订遗漏已被消除；历史块缓存已严格固定预算；主代码已实现论文全部稀疏路径 |

这些论文数字与原稿 A/B/C 的拟议验收阈值性质不同：前者是论文条件下的报告，后者没有实验依据。原稿 B 的“约 1.5 倍有效容量”只是拟议目标，没有先验保证。所有引用都不能拿来预测本项目会提高多少百分点或多少倍有效容量。

## 4. 评测与数据的一手入口

| 来源 | 入口 | 使用边界 |
| --- | --- | --- |
| FineWeb-Edu | [作者数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 语言层训练来源候选；固定数据版本、去重和测试隔离 |
| SlimPajama | [作者数据集](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 训练/适配候选；与原模型预训练分布不同需另报 |
| Zoology / MQAR | [论文](https://arxiv.org/abs/2312.04927) | 可控关联容量与读写诊断；不代表真实Agent任务 |
| RULER | [论文](https://arxiv.org/abs/2404.06654) | 多种长上下文检索/追踪；单针、跨事实与更新题不能混为一个机制 |
| BABILong | [论文](https://arxiv.org/abs/2406.10149) | 长上下文推理补充；先确认短上下文解题能力 |
| LongMemEval | [Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/html/2410.10813)；[作者仓库](https://github.com/xiaowu0162/LongMemEval) | 本会话论文已读、代码入口已定位，未运行；知识更新、时间推理与长期对话记忆适合场景层验证，原始记录可访问的配置不是同持久字节证据 |

自建任务应独立控制事实数、距离、键相似度、修订次数、查询分布、热点及主题漂移。问题在写入后抽取，Oracle归属/图只作诊断；自然语言改写、低频实体与探针覆盖外查询要单列。训练数据、骨干、状态字节与调参机会不能通过跨论文引用代替实际匹配。

## 5. 本目录支持的当前结论

1. 不再将 B 的“多 bank / 非选不衰减 / 内容路由”视为独立新架构；可先试验复用 MoM 的严格总字节干扰反馈调度。
2. 不把 A 的逆矩阵/投影保护公式视为原创空白；重点审查有限选择与合法释放是否有额外价值。
3. 不把 C 的依赖闭包视为充分正确性条件，也不把固定块槽数当成与纯 KDA 等容量。
4. 不把“论文已有机制但没有本项目预算实验”误写成“现有方法没有该机制”。
5. 不用缺少作者代码否定论文全部结果，也不用 README 或独立内核存在证明主推理路径已成熟。
6. 项目名 SteadyMem 已确定；具体算法名随方法与证据明确后确定。先建立任务价值与失败机制；若结构化库、有界精确缓存、RAG或重读更合适，接受这些方案。

## 6. 本轮方法讨论补充的一手来源

以下为 2026 年 10 月 13 日五类方法讨论补充，均没有在本项目复现或部署。

| 来源 | 一手入口与已读范围 | 对候选方法的约束 |
| --- | --- | --- |
| EntNet | [Tracking the World State with Recurrent Entity Networks](https://arxiv.org/pdf/1612.03969)，已读固定动态记忆、key、因果更新与读取机制 | 实体状态维护、无关单元保持和历史后接收问题已有先例；不同任务和训练选择不能当作同字节 KDA 证据 |
| KV-MemNN | [Key-Value Memory Networks for Directly Reading Documents](https://aclanthology.org/D16-1147.pdf)，已读键值编码与 subject/relation/object 表示 | 寻址角色与值解耦已有；保留显式记录不是有界压缩状态的同预算结果 |
| DAMP | [Decay-Aware Mixed-Precision Recurrent-State Quantization](https://arxiv.org/html/2608.27513)，已读状态量化、离线通道校准及混合精度机制 | 优先精度分配已有；不是新事实绑定或版本规则，节省字节后的容量重分配需另做实验 |

这些来源支持 [路线文档第 7 节](ROUTES.md) 的方法边界。稳定寻址、完整损伤控制、干扰反馈、有效性遗忘和当前值覆盖槽均是本项目候选假设，不能因找到近邻就宣称有效，也不能因邻近方法存在就宣布目标问题已经完全解决。
