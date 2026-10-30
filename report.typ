#import "@preview/easy-paper:0.2.2": *

#let body-serif = (
  (name: "New Computer Modern", covers: "latin-in-cjk"),
  "Songti SC",
)

#show: project.with(
  title: "基于颜色直方图与深度特征的以图搜图实验",
  author: "",
  date: auto,
  abstract: [
    #set text(font: body-serif)
    本文实现了基于内容的图像检索（CBIR）完整流程，基于 Imagenette2 数据集（检索库 9,469 张，从验证集中每类选取 2 张共 20 张作为查询图）对比了 48 维 RGB 三通道归一化颜色直方图与 2048 维预训练 ResNet-50 全局池化特征的检索效果。所有特征经 $L_2$ 归一化后采用余弦相似度排序并返回 Top-5 结果。实验结果显示，RGB 颜色直方图的 Top-1 准确率为 45.0%、Precision\@5 为 35.0%，仅在背景色调单一的类别上表现较好，在复杂背景或低饱和度图像上误检较多；预训练 ResNet-50 特征在全部 20 张查询图上的 Top-1 准确率与 Precision\@5 均达到 100.0%。最后结合实验结果讨论了两种特征各自表达与忽略的视觉信息，并给出了面向实例级检索的改进思路与评价指标。
  ],
  keywords: ("以图搜图", "颜色直方图", "ResNet-50", "特征提取"),
  config: (
    title-font: ("New Computer Modern", "Heiti SC"),
    author-font: body-serif,
    body-font: body-serif,
    heading-font: ("New Computer Modern", "Heiti SC"),
    caption-font: ("New Computer Modern", "Kaiti SC"),
    header-font: ("New Computer Modern", "Kaiti SC"),
    strong-font: ("New Computer Modern", "Heiti SC"),
    emph-font: ("New Computer Modern", "Kaiti SC"),
    raw-font: ("Menlo", "Heiti SC"),
  ),
)

#set text(font: body-serif)

= 实验概述

基于内容的图像检索（Content-Based Image Retrieval, CBIR）直接根据查询图像的视觉内容，在图像库中查找相似或同类的图片。由于原始像素维度高且对平移、缩放和视角变化敏感，通常需要先将图像映射为低维特征向量再进行相似度比对。

本实验实现的检索流程包含四个部分：
1. *图像读取与预处理*：统一色彩通道与方向，并按特征提取器的要求完成尺寸与数值变换；
2. *特征提取*：分别提取 RGB 颜色直方图特征与预训练 ResNet-50 深度特征，并做 $L_2$ 归一化；
3. *相似度计算与排序*：计算查询特征与检索库特征的余弦相似度，按得分降序排列；
4. *结果可视化与对比分析*：展示每张查询图的 Top-5 检索结果，统计定量指标并分析典型成功与失败案例。

= 数据集与图像预处理

== 数据集划分

实验直接采用公开数据集 Imagenette2@imagenette（从 ImageNet 抽取的 10 个语义类别子集，涵盖动物、乐器、日常电子设备、工具器械、车辆与建筑户外场景）。项目中 `dataset/gallery` 目录直接使用 Imagenette2 完整的 `train` 划分（共 9,469 张图片）作为检索库，`dataset/query` 目录直接使用其完整的 `val` 划分（共 3,925 张图片）。评测时从 `dataset/query` 的每个类别中固定选取 2 张图片组成共 20 张图片的实验查询集，查询图与检索库无重叠。各类别图片数量见 @tab-dataset，代表样本见 @fig-samples。

#figure(
  table(
    columns: (1.5fr, 1.15fr, 1.1fr, 1fr, 1.1fr),
    align: (left, left, center, center, center),
    table.header([*类别目录名*], [*中文名称*], [*`query` 目录数*], [*实验查询数*], [*`gallery` 数量*]),
    [`English springer`], [英国史宾格犬], [395], [2], [955],
    [`French horn`], [圆号], [394], [2], [956],
    [`cassette player`], [磁带播放机], [357], [2], [993],
    [`chain saw`], [油锯], [386], [2], [858],
    [`church`], [教堂], [409], [2], [941],
    [`garbage truck`], [垃圾车], [389], [2], [961],
    [`gas pump`], [加油泵], [419], [2], [931],
    [`golf ball`], [高尔夫球], [399], [2], [951],
    [`parachute`], [降落伞], [390], [2], [960],
    [`tench`], [丁鲷], [387], [2], [963],
    [*合计*], [—], [*3,925*], [*20*], [*9,469*],
  ),
  caption: [Imagenette2 数据集各类别在 `query` / `gallery` 目录中的图片数量分布],
) <tab-dataset>

#figure(
  image("results/notebook/dataset_samples.png", width: 100%),
  caption: [数据集 10 个类别的代表样本及原始尺寸],
) <fig-samples>

== 图像读取与预处理

数据集中同时包含彩色图、单通道灰度图以及长宽比不同的图片，程序在提取特征前执行如下预处理：

1. *基础读取*：使用 Pillow 读取图像，调用 `ImageOps.exif_transpose` 校正 EXIF 旋转方向，并通过 `.convert("RGB")` 统一转换为三通道 8 位 RGB 图像（像素取值范围 $[0, 255]$）。
2. *颜色直方图预处理*：保留完整原图，不进行裁剪，统计各亮度区间像素频数后除以总像素数 $H times W$，消除图像分辨率差异。
3. *深度特征预处理*：使用 `ResNet50_Weights.IMAGENET1K_V2.transforms()` 提供的标准变换，先按原比例将短边缩放至 232 像素，中心裁剪为 $224 times 224$，再将像素值缩放到 $[0, 1]$ 并按 ImageNet 均值 $mu = [0.485, 0.456, 0.406]$ 和标准差 $sigma = [0.229, 0.224, 0.225]$ 进行标准化。

= 特征提取与相似度计算

== RGB 颜色直方图特征

设输入 RGB 图像尺寸为 $H times W$，将每个通道的灰度范围 $[0, 256)$ 等分为 $B = 16$ 个区间，第 $b$ 个区间（$b = 0, 1, dots, B-1$）对应 $[16b, 16(b+1))$。对每个通道 $c in {R, G, B}$ 统计归一化像素占比：
$ h_c (b) = 1 / (H dot W) sum_(i=1)^H sum_(j=1)^W II (16b <= I_c (i, j) < 16(b+1)) $
将三个通道的 16 维向量拼接为 48 维向量 $bold(h) = [bold(h)_R; bold(h)_G; bold(h)_B] in RR^48$，再进行 $L_2$ 归一化：
$ hat(bold(h)) = bold(h) / norm(bold(h))_2 $

== 预训练 ResNet-50 深度特征

深度特征分支采用在 ImageNet-1K 上预训练的残差网络 ResNet-50@he2016deep（权重版本 `IMAGENET1K_V2`）。ResNet 通过跨层残差连接缓解深层卷积网络的退化问题，其主干提取的高层特征具备较好的跨任务迁移能力。实验中去掉网络末端的 1000 类全连接分类层（`fc`），保留至全局平均池化层（GAP）。输入 $1 times 3 times 224 times 224$ 的标准化图像张量后，在 `eval()` 与 `torch.inference_mode()` 模式下前向计算，将空间尺寸为 $2048 times 7 times 7$ 的特征图经全局平均池化展平为 2048 维向量 $bold(f) in RR^2048$，最后同样做 $L_2$ 归一化得到 $hat(bold(f)) = bold(f) / norm(bold(f))_2$。

单张查询图的预处理图像及两种特征向量的形态见 @fig-feature-illust。

#figure(
  image("results/notebook/feature_illustration.png", width: 100%),
  caption: [单张查询图的 RGB 三通道归一化直方图（48 维）与 ResNet-50 深度特征向量（2048 维）],
) <fig-feature-illust>

== 相似度计算与排序

设 $L_2$ 归一化后的查询图特征为 $hat(bold(q))$，检索库第 $i$ 张图的特征为 $hat(bold(g))_i$，采用余弦相似度计算匹配得分：
$ "sim"(hat(bold(q)), hat(bold(g))_i) = hat(bold(q))^top hat(bold(g))_i $
对检索库全部图片计算相似度后按得分从高到低排序，取前 5 张图片作为 Top-5 检索结果。

= 检索结果与对比分析

== 定量指标统计

在 20 张查询图上统计以下四项指标：
- *Top-1 准确率*：排名第 1 的结果与查询图同类的查询比例；
- *Hit\@5*：Top-5 结果中至少包含 1 张同类图片的查询比例；
- *Precision\@5*：Top-5 结果中同类图片的平均占比；
- *mAP\@5*：在 Top-5 范围内按同类结果出现位置计算的均值平均精度。

整体统计结果见 @tab-eval-overall，分类别 Precision\@5 柱状图对比见 @fig-eval-compare。

#figure(
  table(
    columns: (1.5fr, 0.7fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    table.header([*特征方法*], [*维度*], [*Top-1 准确率*], [*Hit\@5*], [*Precision\@5*], [*mAP\@5*]),
    [RGB 颜色直方图], [48], [45.0%], [85.0%], [35.0%], [49.6%],
    [ResNet-50 深度特征], [2048], [100.0%], [100.0%], [100.0%], [100.0%],
  ),
  caption: [两种特征方法在 20 张查询图上的整体检索指标],
) <tab-eval-overall>

#figure(
  image("results/notebook/evaluation_comparison.png", width: 100%),
  caption: [两种特征方法的（a）整体指标对比与（b）各类别 Precision\@5 对比],
) <fig-eval-compare>

从结果来看，ResNet-50 深度特征在 20 张查询图上的 100 个 Top-5 返回结果全部属于对应类别，四项指标均为 100.0%。RGB 颜色直方图的整体 Precision\@5 为 35.0%，不同类别之间差距明显：背景色调较统一的 `parachute`（70.0%）、`chain saw`（70.0%）和 `golf ball`（60.0%）查准率相对较高；而前景颜色多变或背景占比大的 `French horn`（0.0%）、`cassette player`（10.0%）、`English springer`（20.0%）、`gas pump`（20.0%）和 `tench`（20.0%）误检较多。

== 典型检索案例对比

下面选取 4 张查询图的 Top-5 检索对比图具体说明两种特征的行为差异（图中蓝框为查询图，绿框表示同类，红框表示异类）。

=== 复杂背景干扰（`English springer`）

@fig-case-dog 的查询图为草地上的棕白相间史宾格犬。颜色直方图仅按整图的绿色、棕色与白色像素比例匹配，Top-1 返回了落叶草丛中的白色高尔夫球（相似度 0.9797），Top-3 和 Top-4 分别为树林前的白色垃圾车与树枝间的伐木场景（P\@5 = 1/5）。ResNet-50 提取到了犬类的耳朵、毛色斑块与躯干结构特征，在不同姿态和背景下均检索出同类犬只（P\@5 = 5/5）。

#figure(
  image("results/notebook/compare_01_English_springer.png", width: 100%),
  caption: [`English springer` 查询示例检索对比],
) <fig-case-dog>

=== 低饱和度图像（`church`）

@fig-case-church 的查询图拍摄于浓雾天气，整幅图像接近黑白灰调，R、G、B 三通道分布高度重合。此时颜色直方图主要反映明暗分布，误检出阴影下的油锯和暗色渔网中的丁鲷（P\@5 = 1/5）。ResNet-50 依赖尖塔、拱门等建筑边缘和几何纹理特征，在夜景灯光与晴天光照条件下均正确返回教堂图片（P\@5 = 5/5）。

#figure(
  image("results/notebook/compare_09_church.png", width: 100%),
  caption: [`church` 查询示例检索对比],
) <fig-case-church>

=== 小占比前景目标（`French horn`）

@fig-case-horn 中金色圆号仅占画面中间较小区域，其余大部分像素为深色衣服和室内暗背景。全局颜色直方图受暗色背景像素主导，前 5 名全部返回了同样带有大面积深色区域的油锯、教堂和垃圾车（P\@5 = 0/5）；ResNet-50 则根据盘绕金属管体和号口局部结构找回了 5 张圆号图片。

#figure(
  image("results/notebook/compare_03_French_horn.png", width: 100%),
  caption: [`French horn` 查询示例检索对比],
) <fig-case-horn>

=== 单一纯色背景（`parachute`）

@fig-case-parachute 中两种方法的 P\@5 均为 5/5。但观察颜色直方图返回的 Top-3 结果图可以看到，画面中只有蓝天背景下的跳伞人员，并未出现展开的降落伞伞盖。这说明直方图在该类别上的高命中率主要来自大面积蓝色天空背景的匹配，而非识别出了降落伞主体；ResNet-50 返回的前 5 张图片则均包含完整的伞盖与伞绳结构。

#figure(
  image("results/notebook/compare_17_parachute.png", width: 100%),
  caption: [`parachute` 查询示例检索对比],
) <fig-case-parachute>

= 分析与讨论

== 两种特征描述与忽略的信息

两种特征所保留和丢失的信息对比汇总于 @tab-feature-info。

#figure(
  table(
    columns: (1.45fr, 2.1fr, 2.35fr),
    align: (left, left, left),
    table.header([*特征类型*], [*主要描述的信息*], [*容易忽略的信息*]),
    [RGB 颜色直方图],
    [整幅图像在 R、G、B 三通道各亮度区间的像素占比，反映全局色调与明暗分布。],
    [像素的空间位置关系、前景与背景区分、边缘轮廓、局部纹理及物体语义；对光照和色温变化敏感。],
    [ResNet-50 深度特征],
    [经多层卷积提取的边缘纹理、局部部件与高层类别语义信息，对背景杂波、视角和光照变化较稳健。],
    [精确的空间几何坐标（受下采样和全局平均池化影响）、同类不同个体之间的细微外观差异，以及精确的全局颜色比例。],
  ),
  caption: [两种特征描述与忽略的视觉信息汇总],
) <tab-feature-info>

具体而言：
1. *RGB 颜色直方图*：只对全图像素按颜色区间做频数统计，计算简单且对图像平移、旋转不敏感。但由于统计时丢弃了像素坐标，空间结构、边缘形状和纹理信息全部丢失，无法区分前景物体与背景区域；当前景较小时特征主要由背景决定，且同一物体在不同光照或色温下直方图偏移较大。
2. *ResNet-50 深度特征*：通过层次化卷积逐步组合局部纹理与部件结构，主要刻画图像中包含的物体类别属性。但由于末尾采用全局平均池化（GAP）将 $7 times 7$ 特征图求均值压缩为一维向量，物体的精确空间位置和几何排布信息被弱化；同时，分类任务预训练使网络侧重于拉开类间距离、缩小类内距离，因此同一类别下不同个体之间的细粒度纹理、局部标记或具体配色容易被忽略。

== 面向“找出同一件物品的不同图片”的调整与评价设计

当任务从类别级检索转为实例级检索（找出同一件具体物品在不同角度、光照或背景下的照片）时，模型不仅需要识别物品大类，还需要区分同类别或同款式不同个体之间的局部细节差异，同时消除跨场景背景变化的影响。

=== 检索方法调整

1. *前景检测与裁剪*：不同拍摄场景下的背景差异会干扰全局特征匹配。可在预处理阶段加入目标检测或分割模型（如 YOLO、SAM），定位并裁剪出前景主体区域后再提取特征。
2. *保留局部特征与空间几何验证*：
   - 将全局平均池化替换为对局部显著激活更敏感的 GeM（Generalized Mean Pooling）或多区域池化 R-MAC；
   - 引入两阶段检索：先用全局特征初筛出 Top-$M$（如 Top-50）候选图，再提取局部关键点与描述子（如 SIFT 或 SuperPoint + LightGlue），使用 RANSAC 估计单应性矩阵或基础矩阵，根据几何一致的内点数量对候选结果重排序。
3. *实例级度量学习与颜色信息补充*：
   - 分类预训练权重对类内个体区分度有限，可在实例级数据集（如 Stanford Online Products、Google Landmarks）上采用 Triplet Loss 或 ArcFace 进行度量学习微调，将同一物品的不同视角图作为正样本，同类别不同个体作为难负样本，或直接选用自监督预训练骨干（如 DINOv2）；
   - 针对外观结构相同但颜色不同的同款物品，可在前景掩码区域内融合 HSV 或 Lab 空间的局部颜色特征。

=== 评价标准设计

1. *数据集标注方式*：按独立物品实体划分唯一实例编号（Instance ID），每个实例收集多视角、多光照、不同背景及部分遮挡下的若干张照片，并在检索库中加入同类别其他个体作为干扰项。
2. *评价指标*：
   - *mAP（Mean Average Precision）*：设查询图 $q$ 在检索库中共有 $M_q$ 张同一物品的正样本图片，单次查询的平均精度 $"AP"(q)$ 与整体 $"mAP"$ 定义为：
     $ "AP"(q) = 1 / M_q sum_(k=1)^N P_q (k) dot "rel"_q (k), quad "mAP" = 1 / (|Q|) sum_(q in Q) "AP"(q) $
     其中 $"rel"_q (k) in {0, 1}$ 表示第 $k$ 个结果是否为同一件物品，$P_q (k)$ 为前 $k$ 个结果中的正样本比例，可同时反映查全率与正样本排位靠前程度。
   - *Recall\@$K$（如 R\@1、R\@5、R\@10）与 PR 曲线*：统计前 $K$ 个结果中找回同件物品正样本的比例，并绘制不同截断位置下的 Precision-Recall 曲线。
   - *开集场景下的误识率与拒识率*：若检索库中可能不存在查询物品，需设置相似度阈值 $tau$ 判断“库中无同件物品”，并统计误识率（FAR）与拒识率（FRR）。

#bibliography("ref.bib", title: "参考文献")
