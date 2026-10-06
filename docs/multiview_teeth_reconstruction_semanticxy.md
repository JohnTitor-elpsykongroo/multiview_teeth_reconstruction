> Git 同步版本：本文件复制自项目外部设计文档，后续跨机器修改以本仓库文件为准。用户已授权目标机 Codex 自主推进 R0–R4，当前执行入口为 [自主实验指南](AUTONOMOUS_R0_R4_RESEARCH_GUIDE.md)，命令参考 [分轮训练计划](TRAINING_ITERATION_PLAN.md)。

# 多视图牙齿照片驱动的先验模型 3D 重建：从 SemanticXY 到 Prior Regularization



> 基于论文 **《Teeth Reconstruction and Performance Capture Using a Phone Camera》** 的第 4.2 节，以及围绕“多张牙齿照片 → DMM/牙齿先验 → 3D牙齿重建”的讨论整理。

>

> 本文不是对原论文逐字复述，而是将论文中的关键机制与我们计划中的静态多视图牙齿重建任务重新组织成一条更清晰的技术路线。



---



> 2026-10-06 修订：当前第一阶段为**已知相机、静态上下颌、显式牙位存在性、逐牙语义掩码驱动的合成重建**。上下颌分别训练先验，图像拟合时冻结网络。第六至十一章更新为当前源码与正式训练前的实施计划；旧上颌 sanity 结果保留为历史证据，不自动升级为新双颌验收。

>

> 本文区分“源码接口实现”“小规模开发实验”“正式先验训练”“训练后重建验收”。可运行、导出模型包或单元测试通过都不等于重建质量达标。



> 训练交接更新（2026-10-06）：参考网络全零输出导致的初始法线梯度尖峰已完成根因对照和修复。新配置使用非零 SIREN 输出初始化、`normal_epsilon=0.001`、显式裁剪；冻结数据中的原始 specs 不改写。最新训练配置为 `configs/training_handoff_v1/`，RTX 5090 / WSL2 搬运、预检、smoke、pilot 和长训入口见 [训练交接说明](TRAINING_HANDOFF_20261006.md)。
>
> 本机完成双颌完整输入校验、各一个真实病例 20 步优化，以及各 2 病例/1 epoch 的原生训练链路冒烟。当前短预算验证尚无合格模型，不能直接认定正式训练或重建质量通过；32 病例 pilot 和目标 RTX 5090 实机检查尚待执行。

## 目录



- 第一章 研究目标与总体思路

- 第二章 论文 4.2 的核心机制

- 第三章 SemanticXY：长距离语义对应

- 第四章 Prior Regularization：为什么需要统计先验

- 第五章 其他论文机制：Contact 与 Occlusion

- 第六章 静态多视图牙齿重建方案

- 第七章 工程实施路线

- 第八章 验证策略与消融实验

- 第九章 第一代系统结构与研究核心

- 第十章 后续需要进一步明确的问题

- 第十一章 当前阶段工程优先级

- 第十二章 核心理解总结



---



## 第一章 研究目标与总体思路



目标可以先收缩成：





$$

\boxed{

\text{同一个人的多张 RGB 牙齿照片}

\rightarrow

\text{个体化 3D 牙列}

}

$$



这里不先复现论文完整的“人脸 + 动态表情 + 牙齿 performance capture”，而是只抽取其中对静态牙齿重建最有价值的部分。



论文的 Phase 1 本身就非常接近这个问题：



- 被试保持一个基本固定的露齿状态；

- 手机围绕被试移动；

- 获取多个视角；

- 多帧共享身份相关参数；

- 利用这些多视角观测共同优化牙齿形状与位置。



因此，我们可以把论文完整流程：





$$

z

\rightarrow

DMM

\rightarrow

SDF

\rightarrow

Mesh

\rightarrow

FLAME

\rightarrow

Differentiable\ Renderer

$$



先简化成：





$$

\boxed{

z

\rightarrow

Dental\ Prior

\rightarrow

3D\ Teeth

\rightarrow

Differentiable\ Renderer

}

$$



第一阶段暂时去掉 FLAME、动态 jaw tracking、表情参数和脸部动态遮挡策略；保留双颌/牙龈静态遮挡，并使用：



- 牙齿先验模型；

- 多视图相机；

- 逐牙 semantic mask；

- SemanticXY；

- prior regularization；

- 可微渲染与反向优化。



---



# 第二章 论文 4.2 的核心机制



## 2.1 Analysis-by-Synthesis：整体优化思想



论文 4.2 的核心不是“用一个神经网络直接从图像预测 3D 牙齿”，而是：





$$

\text{当前 3D 参数}

\rightarrow

\text{渲染 2D 结果}

\rightarrow

\text{和真实照片比较}

\rightarrow

\text{计算 Loss}

\rightarrow

\text{反向传播修改 3D 参数}

$$



这就是典型的 **Analysis-by-Synthesis**。



可以把它理解成：



> 先假设一套 3D 牙齿参数，看看从相机里渲染出来像不像真实照片；如果不像，就根据图像误差反过来调整 3D 模型。



---



## 2.2 原论文中的优化变量



原论文中同时存在人脸与牙齿模型，因此优化变量比较多。



牙齿形状：

$$

z

$$

牙齿在 FLAME 人脸坐标系中的刚性安装位置：

$$

M_{up},\quad M_{low}

$$

FLAME 参数：

$$

\beta,\theta,\psi

$$

其中可以粗略理解为：

$$

\beta=\text{身份/脸型}

$$



$$

\theta=\text{头部、下颌等 pose}

$$



$$

\psi=\text{表情}

$$



以上是与几何重建最直接相关的参数。论文的完整目标还包含外观、光照、面部网格偏移等变量与损失；这里没有把它们列为第一版牙齿重建的参数。



所以从牙齿几何拟合的角度，核心是在寻找：





$$

\boxed{

\beta,\theta,\psi,z,M_{up},M_{low}

}

$$



使得渲染出来的人脸和牙齿与真实视频尽可能一致。



而我们的静态牙齿重建第一版可以大幅简化。



---



## 2.3 Semantic Mask Supervision



### 2.3.1 为什么不能只用 RGB



牙齿之间：



- 颜色接近；

- 纹理相似；

- 高光强；

- 相邻牙齿容易混在一起。



因此仅靠 RGB 很难告诉优化器：



> 当前模型里的这颗牙到底应该对应真实图像里的哪颗牙。



论文利用 DMM 自带的 per-tooth semantics，将每颗牙作为不同的语义组件。



例如：



```text

11 → 上颌右中切牙

12 → 上颌右侧切牙

13 → 上颌右尖牙

21 → 上颌左中切牙

22 → 上颌左侧切牙

23 → 上颌左尖牙

...

```



于是对于真实图像中的每个像素，可以定义语义向量：

$$

S(x,y)

$$

例如某像素属于某颗牙：





$$

S(x,y)

=

[0,0,1,0,\ldots]

$$



论文的渲染端并非简单地给每个 mesh 顶点指定一个硬 FDI 标签。它根据 DMM 各牙齿组件的 blending weights，为顶点构造逐牙**软语义向量**；非牙齿区域的向量为零。目标图像则只包含该视角被标注为可见的牙齿通道，通道数 $m_{vis}$ 可随视角变化。



然后通过可微渲染器 $\hat S=\Pi(M,S)$，得到模型在当前相机下的 semantic mask。硬 FDI 标签可以用于最初的简化实验，但须检查其与完整 DMM 混合表面的对应关系。



普通 semantic loss：





$$

L_{sem}

=

\|S-\hat S\|_1

$$



直觉上就是：



> 真实图像中 11 号牙在哪里，模型渲染的 11 号牙也应该出现在那里。



---



## 2.4 普通 Semantic Loss 的局限



假设真实牙齿在图像右边，而模型初始化在很远的左边：



```text

真实：



                    ████





模型：



████

```



此时两块区域完全不重叠。



普通逐像素 loss 看到的是：



```text

左边：

模型有牙

真实没牙



右边：

真实有牙

模型没牙

```



但它很难直接产生：



> “把整颗牙向右移动 200 pixel”



这种稳定、长距离的优化方向。



这就是 SemanticXY 被提出的原因。



---



# 第三章 SemanticXY：长距离语义对应



## 3.1 SemanticXY 的基本构造



SemanticXY 的核心思想可以先压缩成一句话：



> 不仅告诉优化器“这个区域属于哪颗牙”，还告诉它“这个区域在当前三维模型投影后位于哪里”。



普通 semantic supervision 只包含：



$$

S

$$



其中：



$$

S(x,y)

$$



表示图像像素属于哪一个牙齿类别。



例如：



$$

S(x,y) = [0,0,1,0,\dots]

$$



表示该像素属于某颗特定牙齿。



但是，仅有 semantic 信息存在一个问题：



如果模型初始化位置与真实牙齿位置差距很大：



```text

真实：



              ████





模型：



████

```



那么逐像素 semantic loss：



$$

L_{sem} = \|S-\hat S\|_1

$$



只能看到：



- 模型当前位置错误；

- 目标位置缺失；



但无法直接告诉优化器：



> “整颗牙应该向哪个方向移动”。



因此论文引入 SemanticXY。



---



### 3.1.1 什么是 $U(M)$



SemanticXY 在 semantic label 基础上增加屏幕空间位置信息。



对于当前三维牙齿 mesh：



$$

M

$$



定义：



$$

U(M)

$$



表示 mesh 表面点经过当前相机投影后对应的 screen-space XY 坐标。



也就是说：



$$

M \rightarrow U(M)

$$



表示：



> 三维牙齿表面上的点，在当前相机下应该投影到图像哪个位置。



然后通过 differentiable renderer：



$$

\hat U = \Pi(M,U(M))

$$



得到渲染后的 XY map。



最终，将 semantic 和 XY 拼接：



$$

SU=[S,U]

$$



作为高维 point feature。



直观理解：



普通 semantic：



$$

[\text{tooth 11}]

$$



SemanticXY：



$$

[\text{tooth 11},x,y]

$$



也就是：



> “这是 11 号牙，并且它当前投影在图像坐标 $(x,y)$”。



---



### 3.1.2 SemanticXY 的梯度来源



这里有一个重要区别：



SemanticXY 不是因为“加入了像素 XY”就自动产生几何梯度。



真正产生梯度的是：



$$

U(M)

$$



因为：



$$

U

$$



是由三维 mesh：



$$

M

$$



计算得到的。



因此：



$$

\frac{\partial U}{\partial M}\neq0

$$



优化链路为：



$$

M \rightarrow U(M) \rightarrow \hat U \rightarrow L_{semXY} \rightarrow M

$$



随后如果 mesh 来自 DMM：



$$

z \rightarrow DMM \rightarrow M

$$



还需要通过对应的 representation adaptation 方法，将 mesh 上的梯度继续传递回 implicit SDF 和 latent。



因此 SemanticXY 本身解决的是：



$$

\boxed{\text{二维语义对应问题}}

$$



而不是自动解决：



$$

\boxed{\text{implicit model 的梯度传播问题}}

$$



---



## 3.2 为什么 SemanticXY 能提供长距离优化方向



假设真实目标：



$$

11_{target}

$$



位于：



$$

(500,300)

$$



当前模型：



$$

11_{render}

$$



位于：



$$

(300,300)

$$



普通 semantic loss：



只知道：



```text

这里没有牙

那里多了一颗牙

```



但是 SemanticXY + matching 可以建立：



$$

(300,300,\text{tooth11}) \leftrightarrow (500,300,\text{tooth11})

$$



因此优化器可以获得指向目标位置的长距离对应关系。这里的 $\Delta x=200$ 只是直觉示意；最优传输得到的是软匹配，不保证每个像素都形成精确的同牙硬对应或恰好移动 200 pixel。



于是：



$$

L_{semXY}

$$



能够提供：



> “模型牙齿应该向右移动”的全局优化方向。



---



## 3.3 Optimal Transport 在 SemanticXY 中的作用



论文不是直接比较：



$$

(x,y)\rightarrow(x,y)

$$



而是首先建立：



$$

Warp(\hat{SU},SU)

$$



对应关系。



论文在建立 $Warp(\hat{SU},SU)$ 后，定义



$$

L_{semXY}=\|Warp(\hat{SU},SU)-\hat{SU}\|_1

$$



其中 $SU=[S,U]$ 包含：



- semantic similarity；

- spatial proximity。



因此 OT 的作用是：



> 在可能存在较大空间偏移时，寻找语义一致区域之间的对应关系。



例如：



真实：



```text

        11   21

      12       22

```



当前模型：



```text

11   21

12       22

```



SemanticXY + OT：



```text

render 11  -------> target 11



render 21  -------> target 21



render 12  -------> target 12



render 22  -------> target 22

```



然后利用这些 correspondence 优化三维位置。



因此：



$$

\boxed{L_{semXY} = \text{负责建立长距离对应}}

$$



而：



$$

\boxed{L_{sem} = \text{负责局部精细对齐}}

$$



两者共同作用。这里的“长距离”和“局部”是便于理解的功能侧重，不代表两个 loss 严格分工；它们都可能影响最终对齐。



---



# 第四章 Prior Regularization：为什么需要统计先验



## 4.1 论文中的 Regularization



SemanticXY 和 semantic loss 都只关心：



> 渲染出来像不像照片。



但这里会出现另一个问题：



> 一个 3D 模型投影得很像，并不代表它的 3D 几何一定合理。



因此论文增加了 prior regularization。



论文使用：





$$

L_{\text{prior,teeth}}

=

\lambda_{\text{teeth}}

\|z\|_2^2

$$



目的是让 DMM latent code 不要为了迎合图像误差跑到非常极端的区域。



---



## 4.2 为什么多视图仍然需要 Prior



假设只有一张正面照片。



下面两种 3D 牙齿可能正面投影非常接近：



```text

方案 A：



      正常牙齿

     _______

    /       \

   |         |

    \_______/





方案 B：



      奇怪牙齿

     _______

    /       \

   |         |================

    \_______/



正面投影可能差不多，

但方案 B 的深度方向已经严重畸变。

```



所以：





$$

\Pi(T_A)

\approx

\Pi(T_B)

$$



导致：





$$

L_{image}(T_A)

\approx

L_{image}(T_B)

$$



多张照片可以显著减少这种歧义，但无法完全消除，因为现实照片仍然无法完整看到：



- 牙齿背面；

- 邻接面；

- 很多后牙区域；

- 牙根；

- 牙龈以下结构；

- 被遮挡部分。



因此真正要解决的不是 $\text{哪个 3D 模型投影最像？}$；而是：





$$

\boxed{

\text{哪个“合理的牙齿模型”投影最像？}

}

$$



“合理”就是 prior 提供的。



---



## 4.3 Latent Code $z$ 的直觉



假设牙齿先验模型为 $T(z)$，其中 $z=(z_1,z_2,\ldots,z_n)$。可以把这些 latent dimensions 暂时想象成很多“形变旋钮”：



```text

z1：牙弓整体宽窄变化

z2：门牙突出程度

z3：牙冠高度变化

z4：某类牙齿尺寸变化

z5：牙列局部排列变化

...

```



真实 neural latent 未必具有如此清晰的单独语义，但直觉类似。



上述各维对应具体形变的说法只是虚构的全局 latent 示例。当前 DMM 使用牙龈和逐牙独立编码，不能给任一维直接赋予这些物理语义。论文的零中心 L2 倾向于让 $z$ 靠近零，但零码是否对应经验平均形状、正常变化尺度是多少，仍须由实际训练码和解码结果核对；数值大小不能照搬其他模型。



---



## 4.4 $\|z\|^2$ 到底在做什么



论文的 prior：





$$

L_{prior}

=

\lambda_z\|z\|_2^2

$$



展开：





$$

L_{prior}

=

\lambda_z

(z_1^2+z_2^2+\cdots+z_n^2)

$$



例如 $z=[0.2,0.3]$，则 $\|z\|^2=0.13$，很小。



如果 $z=[5,7]$，则 $\|z\|^2=74$，非常大。



所以它是在告诉优化器：



> 可以改变牙齿，但没有充分图像证据时，不要离正常分布中心太远。



---



## 4.5 图像 Loss 与 Prior 的权衡



假设：





$$

L

=

L_{image}

+

\lambda L_{prior}

$$



其中：





$$

L_{image}

=

L_{sem}

+

L_{semXY}

$$



图像 loss 在说：



> “你的牙齿投影和照片不像，继续变。”



prior 在说：



> “可以变，但别为了少几个像素误差变得过分异常。”



最终得到的是：





$$

\boxed{

\text{既能解释照片，又符合牙齿统计规律的解}

}

$$



---



### 4.5.1 一个数字例子



候选解 A 的图像误差为 $L_{image}=1$，但 $\|z\|^2=100$。



若 $\lambda=0.1$，则





$$

L_A

=

1+0.1\times100

=

11

$$



候选解 B 的图像误差为 $L_{image}=2$，且 $\|z\|^2=4$，则





$$

L_B

=

2+0.1\times4

=

2.4

$$



虽然 A 在二维照片上更“完美拟合”，但 B 的 3D 结构更加可信，因此总 loss 更低。



---



## 4.6 PCA 先验中的 $z^T\Sigma^{-1}z$



如果我们的牙齿先验不是论文的 neural DMM，而是 PCA/statistical model，就可以进一步利用各主成分的统计方差。



假设：





$$

x

=

\bar x

+

P\alpha

$$



其中：



- $\bar x$：平均牙列；

- $P$：PCA basis；

- $\alpha$：PCA coefficient。



训练数据还会提供每个 PCA 方向的 eigenvalue $\lambda_1,\lambda_2,\ldots,\lambda_d$。如果 $\lambda_1=100$，说明 PC1 在真实人群里变化很大。



如果 $\lambda_2=1$，说明 PC2 在真实人群里变化很小。



即使 $\alpha_1=5,\quad \alpha_2=5$，它们的统计意义也完全不同。



---



### 4.6.1 为什么不能简单一视同仁



普通 L2：





$$

\|\alpha\|^2

=

\alpha_1^2+\alpha_2^2

$$



会认为两个方向完全一样。



但更合理的是：





$$

L_{prior}

=

\sum_i

\frac{\alpha_i^2}{\lambda_i}

$$



即：





$$

L_{prior}

=

\alpha^T\Sigma^{-1}\alpha

$$



如果：





$$

\Sigma

=

\begin{bmatrix}

100&0\\

0&1

\end{bmatrix}

$$



那么：





$$

\Sigma^{-1}

=

\begin{bmatrix}

1/100&0\\

0&1

\end{bmatrix}

$$



对于 $\alpha_1=\alpha_2=5$，得到：





$$

L_{prior}

=

\frac{25}{100}

+

25

=

25.25

$$



其中 $PC1:0.25$，而 $PC2:25$。所以：



> 真实人群本来变化大的方向，可以允许多走一些；



> 真实人群几乎不变化的方向，一旦走得太远就要受到更强惩罚。



---



## 4.7 Prior 本质上是在计算“偏离几个标准差”



因为：

$$

\lambda_i=\sigma_i^2

$$

所以：





$$

\frac{\alpha_i^2}{\lambda_i}

=

\left(

\frac{\alpha_i}{\sigma_i}

\right)^2

$$



也就是说：



> prior 实际上在衡量当前参数离正常人群平均值多少个标准差。



如果 $\alpha_i=1\sigma$，惩罚约为 $1$，如果 $\alpha_i=3\sigma$，惩罚约为 $9$，如果 $\alpha_i=10\sigma$，惩罚约为：$100$。



---



## 4.8 L2 与 Mahalanobis Prior 的关系



如果我们已经把 PCA coefficient whiten：





$$

z_i

=

\frac{\alpha_i}{\sqrt{\lambda_i}}

$$



则 $z$ 的经验均值约为零、协方差约为 $I$。只有原始系数分布近似高斯时，才可进一步近似写成 $z\sim N(0,I)$；白化本身不保证高斯分布。



此时：





$$

\|z\|^2

=

\sum_i

\frac{\alpha_i^2}{\lambda_i}

$$



也就是说：





$$

\boxed{

\|z\|^2

\equiv

\alpha^T\Sigma^{-1}\alpha

}

$$



只不过参数定义不同。



所以以后真正要问的不是：



> 到底用 L2 还是 Mahalanobis？



而是：





$$

\boxed{

\text{我们的 latent code 是怎么定义和标准化的？}

}

$$



如果 latent 已经中心化、白化，且各向同性高斯近似与经验分布相符，那么 $\|z\|^2$ 可作为相应的二次 prior；仅凭白化不能证明它是精确的人群概率。



如果保存的是原始 PCA coefficients $\alpha_i$，那么通常更自然的是：$\sum_i\frac{\alpha_i^2}{\lambda_i}$。



---



## 4.9 为什么有先验模型还需要 Prior Loss



这是最容易混淆的一点。



答案是：





$$

\boxed{

\text{“用了先验模型”}

\neq

\text{“所有可能的参数都一定是合理牙齿”}

}

$$



---



### 4.9.1 PCA 学的是“方向”，不是“边界”



假设：





$$

T(\alpha)

=

\bar T+P\alpha

$$



PCA basis $P$ 是从真实牙齿数据中学习的。



但是 PCA 本身并不会限制 $\alpha_i$，只能待在训练数据范围内。



假设训练数据中的某个主成分大致分布在：





$$

-3

\lesssim

\alpha_1

\lesssim

3

$$



数学上优化器仍然完全可以取：$\alpha_1=20$。



甚至：$\alpha_1=100$。



于是：





$$

T(100)

=

\bar T+100P_1

$$



依然可以计算出来。



但它已经可能对应现实中完全不存在的牙齿。



所以 PCA 告诉我们的是：



> “牙齿主要沿哪些方向变化。”



却没有自动规定：



> “沿这些方向最多走多远。”



---



### 4.9.2 “先验模型”与“Prior Loss”分别解决什么问题



可以把它们明确区分成：





$$

\boxed{

T(z)

=

\text{允许怎么变}

}

$$



而：





$$

\boxed{

L_{prior}(z)

=

\text{哪些变化更可信}

}

$$



前者限制的是：



> 模型的表达空间。



后者限制的是：



> 优化结果在这个空间里的统计位置。



---



### 4.9.3 Neural DMM 同样存在这个问题



论文使用的是 neural SDF-based DMM。



网络会根据 latent code $z$ 以及空间坐标 $x$ 输出各牙齿/牙龈组件的 SDF。



一个容易产生的误区是：



> “这个网络是用真实牙齿训练出来的，所以任意 $z$ 都应该生成正常牙齿。”



实际上并不是。



训练时网络只在训练码覆盖的 latent 区域得到约束。若优化器不受限制，仍可能走到这些区域之外；常见的 $[-2,2]^d$ 或 $[8,-12,15,\ldots]$ 只能作抽象示意，不能当作当前 DMM 的实测范围。



神经网络依然会给出一个 SDF。



但是这是：





$$

\boxed{\text{extrapolation}}

$$



而不是：





$$

\boxed{\text{interpolation}}

$$



网络“能够输出结果”，不代表这个结果在训练分布中具有统计意义。



---



### 4.9.4 为什么图像优化尤其容易把 latent 推远



因为监督实际上发生在 $\Pi(T(z))$，也就是二维投影上。



可能存在 $T(z_1),T(z_2),T(z_3)$，它们满足：





$$

\Pi(T(z_1))

\approx

\Pi(T(z_2))

\approx

\Pi(T(z_3))

$$



但：



- $T(z_1)$ 是正常 3D 牙齿；

- $T(z_3)$ 可能在不可见区域严重扭曲。



因为相机看不到这些异常区域：





$$

L_{image}(z_1)

\approx

L_{image}(z_3)

$$



所以必须额外要求：





$$

L_{prior}(z_1)

<

L_{prior}(z_3)

$$



才能倾向于选择合理 3D 解。



---



### 4.9.5 从概率角度理解 Prior：MAP Estimation



真正的问题并不是：





$$

z^*

=

\arg\min_z L_{image}

$$



而是：



> 给定所有照片 $I$，哪一个牙齿参数 $z$ 最可能？



即：$p(z|I)$。



根据 Bayes：





$$

p(z|I)

\propto

p(I|z)p(z)

$$



其中 $p(I|z)$ 表示 



> 这个 3D 牙齿渲染出来有多像照片。



对应 $L_{image}$。而 $p(z)$ 表示 



> 这种牙齿在人群统计中有多可能。



对应 $L_{prior}$。严格地说，这要求 $p(z)$ 是有依据的分布假设或经数据估计的概率；论文采用的零中心 L2 是一种正则化形式，并不等于已经学得完整的人群牙列概率。



取负对数：





$$

-\log p(z|I)

=

-\log p(I|z)

-

\log p(z)

+

const

$$



于是自然得到：





$$

\boxed{

L

=

L_{image}

+

L_{prior}

}

$$



这就是 Maximum A Posteriori（MAP）估计。



---



### 4.9.6 Prior 不意味着把所有人压回平均牙



假设某个人确实有非常特殊的牙齿形态。



多张照片从不同角度都强烈表明 $z$ 必须远离平均值。



此时即使 $L_{prior}$ 增加，只要：





$$

\Delta L_{image}

\gg

\Delta L_{prior}

$$



优化器仍然会往那个方向走。



所以 prior 的真正意义是：



> 没有足够证据时，相信统计规律；



> 图像证据非常强时，允许个体偏离平均状态。



---



### 4.9.7 什么情况下可以弱化显式 Prior Loss



理论上，如果我们构造了一个带有严格有效域的生成模型 $z\in\mathcal Z$，并且能够保证 $\forall z\in\mathcal Z$，都只生成合理牙齿，那么显式 prior loss 可以变弱。



例如 $z_i\in[-3,3]$，且整个区域都被可靠覆盖。



但现实中的：



- PCA；

- Auto-decoder；

- VAE；

- Neural implicit DMM；



通常都没有这样的硬保证。



因此实际系统仍然常常使用：



- latent bounds；

- prior loss；

- 或二者结合。



---



# 第五章 其他论文机制：Contact 与 Occlusion



## 5.1 Contact Loss



论文除了 latent prior，还使用了 contact loss。



牙齿使用 SDF $\Phi(x;z)$，如果一个嘴唇/面部点进入牙齿内部 $\Phi(x)<0$，则认为发生穿插。



论文的式 (9) 先把面部顶点 $x_v$ 经脸部运动 $W^{-1}$ 和牙列安装变换 $M_*^{-1}$ 送回 DMM 的 canonical 坐标，再查询牙齿 SDF：



$$

L_{contact}=\|\max[-\Phi(M_*^{-1}W^{-1}(x_v)),0]\|_1,\qquad M_*\in\{M_{up},M_{low}\}

$$



如果 $\Phi>0$，点位于牙齿外且该项为零；如果 $\Phi<0$，负值幅度产生穿插惩罚。



它本质上是一个 penetration loss。



不过对于我们第一版静态牙齿重建，如果暂时没有完整人脸/嘴唇模型，可以先不实现。



---



## 5.2 遮挡问题



论文完整系统还处理牙齿从：



```text

可见

→

完全遮挡

```



以及：



```text

完全遮挡

→

重新出现

```



这种动态状态变化。



牙齿一旦完全被嘴唇挡住，可微渲染中的牙齿梯度可能变得很弱甚至消失。



论文设计了两个策略：



- Face-vanishing；

- Teeth-hiding。



---



### 5.2.1 Face-Vanishing



情况：



```text

真实图像：牙齿可见



当前模型：牙齿完全被嘴唇遮住

```



正常 renderer 看不到牙齿，因此没有足够梯度。



论文临时移除遮挡牙齿的脸部区域，再渲染牙齿：$\hat{SU}_{no\ occ}$。



然后计算：





$$

L_{f-v}

=

semXY(

\hat{SU}_{no\ occ},SU

)

$$



它的作用是：



> 把模型牙齿从错误的“完全被遮挡状态”中重新拉出来。



---



### 5.2.2 Teeth-Hiding



相反情况：



```text

真实图像：牙齿完全看不到



模型：牙齿还露在外面

```



此时真实图像里没有牙齿 target 可用于匹配。



论文利用人体结构先验：



> 牙齿看不到时，应该是被嘴唇及周围面部区域挡住。



于是让模型牙齿向应当遮挡它的区域移动：





$$

L_{t-h}

=

semXY(

\hat{SU},

\hat{SU}_{occ}

)

$$



---



## 5.3 对静态多视图任务的取舍



如果某张照片只标注了 $11,12,13,21,22,23$，这一视角的正向逐牙监督只覆盖这些可靠可见的牙齿；其余 FDI 不能因没有标签就判为缺牙。



但“没有标注”和“确认无遮挡且没有牙齿”也不能混为一谈。应区分：



- 被唇颊或其他牙齿遮挡；

- 超出视野或未完成标注；

- 在可靠可见区域确实没有牙齿像素；

- 经过独立确认的实际缺牙。



前两种情形不对对应牙齿施加缺牙惩罚。对于第三种情形，如果模型在确认无遮挡的区域错误地露出牙齿，仍应计入负向图像证据。其他视角和 dental prior 负责约束不可见的 3D 区域；仅靠照片不能证明这些区域的真实个体形态。



---



# 第六章 静态多视图牙齿重建方案



## 6.1 第一阶段的目标、输入与输出



第一阶段限定为：**已知相机、同一静态双颌状态、逐牙语义掩码驱动的合成重建**。默认上下颌各 14 牙及牙龈；不包含智齿，允许具有明确存在性标注的缺牙。静态意味着两颌形态及相对位姿跨视角共享，不意味着两颌世界位姿已知。



输入为 $\mathcal I=\{(C_k,S_k,V_k)\}_{k=1}^N$、两套已训练模型包与明确的 presence。相机固定，RGB 保留用于观测生成及人工检查，本阶段不优化光度损失。标签使用原始 FDI，0 为背景、255 为 ignore，`valid_mask == labels != 255`。图像里没出现的牙不等于缺牙。



输出为两颌共享参数、毫米世界坐标网格、逐牙逐视图拟合诊断及未观察牙位列表。源扫描、深度、真值位姿与真值码只供评价或明确标记的 oracle 诊断；主拟合器不读取它们。



## 6.2 当前 DMM 的参数与共享关系



$$

\Theta=\{M_{up},M_{low},z_{up,i},z_{low,j}\}.

$$



使用两个独立单颌 decoder。每颌一个 SE(3) 整体位姿，所有视角共享。首版牙龈码固定为训练均值；始终保留牙龈几何遮挡。每牙 latent 是耦合形态、排列、空间变形、SDF 修正及混合权重的代码，不称为纯 shape code。查询点相关的 screw field 不等于每牙一个自由刚体位姿。本版不另加逐牙位姿参数。



`Teeth3DS_DualArch_v1` 当前 specs 为牙龈和每牙各 10 维：每个单颌模型含 150 维组件代码；牙龈固定、14 牙全开放时，每颌拟合 140 维 latent 加 6DoF。存在但全视角未见的牙保持初始先验码并报告 unobserved。维数由 bundle 读取，不硬编码。旧 `20260824` 上颌实验的每牙 20 维及其尺度、权重不自动兼容新模型。



模型坐标统一 `1 unit = 50 mm`。训练几何在核实源单位后只做刚体对齐，不进行每病例独立缩放。投影顺序是模型坐标乘 50、施加 $M_{arch}$、再施加世界到相机变换和 K。两套独立先验不等于跨颌咬合先验。



## 6.3 DMM、语义渲染与梯度桥



$$

z\rightarrow\Phi(x;z)\rightarrow T(z)\rightarrow M_{arch}T(z)

\rightarrow\Pi(C_k,\cdot)\rightarrow(\hat S_k,\hat U_k)\rightarrow L.

$$



原 `utils.mesh.create_mesh` 是导出入口，其 detach 后的网格不能直接反传 latent。当前源码已新增 `dmm/surface.py`：MC 提供离散拓扑与锚点，沿固定法线求零点，再以隐函数定理把局部曲面梯度接回 latent。局部导数为 $\partial x/\partial z=-n(\partial\Phi/\partial z)/(\nabla\Phi\cdot n)$；它依赖当前局部曲面与非奇异分母，不宣称离散拓扑变化处全局光滑。



`dmm/rendering.py` 将两颌混合表面放入同一个深度缓冲，使用 DMM blending weights 构造逐牙软语义。牙龈在语义中归于背景，但继续遮挡其他几何。材质点 XY 在每次反向中固定光栅对应，从几何投影获得梯度。CPU 渲染只作诊断；正式图像拟合使用 CUDA 抗锯齿后端。



已有局部梯度和夹具证据；细密 MC 网格整图 AA 的严格有限差分仍存在离散轮廓切换限制。回溯与重提网格后的实际下降检查必须保留。训练后还须在真实牙齿几何上核对梯度与优化行为。



## 6.4 静态条件、相机和像素中心



同一 scene 的双颌状态不随照片改变。相机使用 mm、右手系，+X 向右、+Y 向下、+Z 向前；所有外参属于同一世界坐标。先去畸变，不能把非线性去畸变写成裁剪仿射。



新增显式 `pixel_convention`，保持旧实验可解释：



| 约定 | 像素中心 | 裁剪 x0 后缩放 sx |

|---|---|---|

| `integer_centers`，旧清单缺字段时使用 | $(u,v)=(col,row)$ | $u'=s_x(u-x_0+0.5)-0.5$ |

| `edge_origin_centers_at_half`，新照片入口 | $(u,v)=(col+0.5,row+0.5)$ | $u'=s_x(u-x_0)$ |



两种约定都要求 $K_{fit}=A_{fit\leftarrow source}K_{source}$，但 A 的平移项随约定不同。读取新照片时不得暗加减 K 的 0.5。CPU 像素采样、CUDA clip 投影、SemanticXY、恢复射线均按同一显式约定计算。归一化中心分别为 $((W-1)/2,(H-1)/2)$ 或 $(W/2,H/2)$，分母均为 $\sqrt{W^2+H^2}$。



每颌至少两个不同可见相机是输入下限，不代表每颗牙都可观测。全背景有效视图保留负向证据；全 ignore 视图不能悄悄作为有效视角参与拟合。不同静态张口状态必须拆为不同 scene。



## 6.5 分阶段优化与验证



以下 M0–M4 是重建实验顺序，不是先验网络训练的 epoch。



- **M0 前向**：核对两颌、牙龈、FDI、mm、投影、像素中心和遮挡。

- **M1 Pose-only**：合成诊断可显式固定 GT 几何，仅恢复两颌整体位姿；与生产初始化严格区分。

- **M2 Latent-only**：合成诊断可固定 GT 整体位姿，核对图像到 latent 导数和几何改善，不以唯一恢复 GT code 为指标。

- **M3 Joint**：均值模型或独立估计初始化，冻结 decoder，位姿→latent→交替联合更新；无几何真值输入。

- **M4 Prior/视角扩展**：比较可观测牙、后牙、缺牙及遮挡，记录预留视角和毫米几何误差。



源码 `shape` 阶段实际上为 latent-only。当前 prior 使用 $z_i=\mu_i+L_iq_i$、$R=\operatorname{mean}_i\operatorname{mean}(q_i^2)$。统计只取原始训练病例且对应牙存在的代码，排除镜像、val/test 和缺牙占位项；白化不证明高斯。ICCV 的原始 $\|z\|^2$ 与此不是同一数值目标。



当前 `partial_soft_semanticxy_v1` 使用 28 通道条件前景语义、面积质量、部分 OT/dustbin、冻结对应的双向 Warp、dense 语义及未匹配损失，各视图等权平均。它是论文启发的实现，不是式 (7) 的逐行复刻；采样数和权重是实验配置，不是论文保证的最优值。



## 6.6 后续扩展边界



首版保持两套单颌先验、显式存在性和耦合 latent。跨颌防穿插可选，不强制张开双颌接触，不等同于论文牙—脸 contact。先在裸露双颌闭环，再研究可信软组织 ignore、分割噪声与真实照片。



固定均值牙龈适用于受控目标；独立扫描中的牙龈差异要单独诊断，不能让牙齿 latent 无声补偿。若证据表明有必要，再开放强正则牙龈参数或另立模型版本。逐牙刚体/形态解耦、跨颌联合先验、动态跟踪和相机自标定不作为本轮长训前的强制条件。



---



# 第七章 工程实施路线



## 7.1 数据边界与训练入口



正式输入来自 `D:/WorkSpace/Dental/data/Teeth3DS_DualArch_v1/training/upper|lower/` 的 manifest、canonical 和 specs。`work/` 是缓存，`reports/` 是诊断；不得修改冻结数据来迁就失败训练。单位依据、刚体变换、FDI、presence、患者 split、镜像血缘及资源 SHA256 都必须保留。



当前数据为上颌 train 645 原始/1290 含镜像、val 88、test 80；下颌 train 638 原始/1276 含镜像、val 86、test 78。数据 READY_FOR_TRAINING 只表示数据契约与 QC 就绪，不表示训练配方或重建系统通过。



三维监督是观测到的牙冠/牙龈表面、法线和非表面点，没有牙根或人工封底。DMM 无需数值 signed-distance GT；不能为了适配名字中的 SDF 强行给开口扫描填造内部真值。



## 7.2 训练配方、诊断与对照



训练损失配置化：组件表面、防翻转、非表面、法线、Eikonal、修正场、变形平滑、模板法线、blend、中心、latent 以及整体场项分别保存原值和加权值。记录每组件有效点数、整轮分项均值、分组梯度范数及实际学习率。



有效点均值与全点掩码均值不是同一目标。完整 15 组件各 256 表面点加 2048 非表面点时，同牙表面项分母从 5888 变成 256，名义系数不变就可能相差约 23 倍。中心项的组件分母也须明确。用显式版本记录配方，不能把新目标叫作官方等价训练。



deformation、reference 和 latent 使用独立参数组；学习率调度及 Adam 状态随检查点保存。可配置牙龈采样额度和梯度裁剪，未知配置键/非有限值应拒绝。小规模对照固定病例、种子和预算，先比较归一化、采样与正则；不凭总 loss 绝对值跨配方选优。



`effective_count_v1` 保留本地旧路径默认行为；`masked_mean_v1` 显式采用全点掩码分母和含牙龈的中心项分母。`configs/training_readiness/recipe.json` 是后者的候选配置，另提供 effective 对照；初始化方差、学习率衰减等均显式保存，尚未通过真实训练校准。上下颌 `_pilot.json` 各选择 32 个原始 train 病例、4 个原始 val 病例、2 epoch，供后续短程检查；不是已经完成的训练。`_formal_candidate.json` 仅为完整配置草案。



`training_case_ids` 限定实际参与训练的集合；选入镜像必须包含原病例。导出的 prior 只统计该集合中的原始、存在牙代码，不能混入未训练的占位 embedding。独立 checkpoint 评价要求训练子集和训练配方与 checkpoint 相同，验证采样预算可另行明确设置。



## 7.3 独立 auto-decoder 验证与选模



每个候选 checkpoint 冻结 decoder，为原始 val 病例新建临时 latent，以训练原始存在牙的代码均值初始化。使用固定优化预算和种子，拟合点与评价点按持久点索引分成互斥集合。验证不得更新 decoder、训练 embedding 或训练统计；test 不参与配方调参。



验证拟合的逐牙中心也仅从拟合分区采样点计算，不使用包含评价点的全扫描缓存中心。数据入口仍可校验全部清单及其资源，但 test 不进入优化、选模或打分。



`training/validation.py` 记录逐病例失败、逐牙毫米采样表面距离、法线及语义错误和网格边界情况。当前选模分数为逐牙再逐病例平均的“保留扫描点→混合网格面积采样点”距离；反向距离另记，因为生成表面含没有扫描真值的隐藏部分。此分数受分辨率和采样密度影响，不是精确 Hausdorff，也不单独构成解剖验收。



任何选定验证病例失败，整轮 score 为 null，不丢弃病例算平均、不产生新的 `best_val`。保留 `best_train`、`best_val` 和 `final` 的区别。`case_ids` 若指定，只能选择原始 val，必须在实验开始前冻结；不能事后挑容易病例。



## 7.4 模型包、来源与兼容



每次训练保存完整 Python 源码快照、全源码指纹、配置、数据哈希、优化器/调度器与随机状态。精确续训仍要求同训练源码与配置签名，旧检查点不得默默转换。



训练续跑状态与图像拟合轨迹的重现性应分别验收。图像拟合保存 `starting_parameters.json`，验证 GPU 参数、历史和游标恢复；CUDA 浮点归约及重提取的非光滑网格可能使两个不中断控制也出现轨迹差异。本轮未通过原 GPU 严格轨迹比较诊断，不能写成“GPU 续跑结果完全一致”；确定性参考后端另做严格连续运行对照。



新 bundle 额外记录 decoder 执行契约：网络和必要数学依赖的代码哈希、组件查询语义、固定尺度及统计格式。渲染器/OT/拟合器改动不自动使兼容权重失效；decoder 或尺度变化则拒绝。旧 bundle 没有此字段时继续使用完整源码严格绑定，不能补写一个新哈希假装兼容。



导出只使用原始训练存在牙的代码统计，与同一权重/checkpoint 绑定。成功导出状态仍为 NOT_QUALITY_ACCEPTED。两颌必须各自通过单颌质量检查后才用于主重建实验。



## 7.5 均值初始化与多视图拟合



`dmm/initialization.py` 提供均值模型候选：从多视图可见掩码质心构造射线约束，非退化时提供刚体候选，并加入 24 个轴向旋转候选；在固定尺度下解平移，用两颌共同深度缓冲及所有视图的实际语义目标筛选。



掩码质心不是跨视角同源解剖点，该模块只给启发式初值。共线、零基线、相机后方、模型无表面等情况必须报告。候选成功不是最终姿态恢复或通用大旋转保证；需要实际牙列测试。之后仍运行 pose→latent→joint 和实际下降/重提网格检查。



## 7.6 原生命令与照片接口



新训练使用已经解析并绑定资源哈希的交接配置，不再用冻结数据目录中的旧 specs 生成默认训练配置。以下为本机检查或开发时的入口；目标 WSL 请使用交接说明中的脚本。

```powershell
Set-Location D:/WorkSpace/Dental/multiview_teeth_reconstruction
. ./scripts/gpu_env.ps1
& $gpuPython third_party/DMM/dmm_cli.py validate-training `
  --config configs/training_handoff_v1/upper_pilot.json `
  --other-config configs/training_handoff_v1/lower_pilot.json

# 下一阶段有限病例实验；不是本轮已经执行的长训。
& $gpuPython third_party/DMM/dmm_cli.py train `
  --config configs/training_handoff_v1/upper_pilot.json `
  --output runs/upper_pilot_new --device cuda

# 有有效 best_val 后才执行；不存在时先检查验证失败。
& $gpuPython third_party/DMM/dmm_cli.py evaluate-checkpoint `
  --config configs/training_handoff_v1/upper_pilot.json `
  --checkpoint runs/upper_pilot_new/best_val.pth --output runs/upper_validation_new --device cuda
& $gpuPython third_party/DMM/dmm_cli.py export-bundle `
  --checkpoint runs/upper_pilot_new/best_val.pth --output runs/upper_bundle_new --model-id upper-pilot-v1
```

照片阶段三的旧输出仍是 candidate，不能手工改名绕过旧阻塞。模型和独立初值齐备后，可使用 `prepare-scene <candidate_manifest.json> --output <new_scene_root>`，显式校验半像素约定并复制白名单引用到新拟合包。它不读取 truth，不修改旧候选，fit_ready 只表示输入可消费。



```powershell

& $gpuPython third_party/DMM/dmm_cli.py fit-scene <new_scene_root>/fit_input/manifest.json `

  --device cuda --initialize-mean --fit-config configs/fit_static_v1.json `

  --surface-config configs/surface_static_v1.json --render-config configs/render_nvdiffrast_v1.json `

  --matching-config configs/semanticxy_static_v1.json --output runs/fit_new

```



`--initialize-mean` 从模型均值重新估计候选，不能与 `--resume` 同用。固定相机/存在性仍需明确来源。图像 confidence 目前保存但不作为像素损失权重，不能把侧车文件存在解释成加权拟合已实现。



## 7.7 后续 RGB 与软组织扩展



先使用可靠逐牙掩码分离几何问题，再研究 RGB→掩码误差。裸露双颌与唇舌遮挡分开验收；采集 valid 不代表牙齿拟合的可信背景。GT tissue/GT mask 只可在显式 oracle 诊断使用，不冒充预测结果。



---



# 第八章 验证策略与消融实验



## 8.1 三类证据分开报告



| 证据 | 支持什么 | 不支持什么 |

|---|---|---|

| 接口、CPU/CUDA、局部梯度与短训夹具 | 实现、导数局部行为、save/resume/export 流程 | 真实牙冠精度、训练泛化 |

| DMM 生成目标的同模型合成恢复 | 当前表示内部的优化恢复 | 独立扫描的表示能力 |

| 保留患者扫描渲染后的重建 | 模型表示与图像拟合的联合能力 | 未扫描牙根、真实照片或真实咬合真实性 |



旧 sanity 结果保持原始范围和状态，新源码不复用其通过标记。训练用三维数据与分割用 RGB 数据是两个不同问题，不因已有照片而认为必须端到端联合训练。



## 8.2 训练前、训练后两道关口



训练前要求：契约明确、完整配置可验证、有限值及有效点统计可追踪、独立验证隔离、精确续跑、导出/加载和梯度链路具备针对性证据。通过有限病例过拟合和短程独立验证后才决定正式训练配方。



训练后要求：上下颌各自具备可用混合表面、逐牙语义和稳定先验统计，再做均值初始化、位姿/latent/联合恢复。保存逐牙逐视图 IoU、预留视角、观察面与未观察面误差、world-mm 位姿/表面误差、所有失败及不可观测牙。



主评价不能用事后自由刚体配准消除位姿错误。额外对齐后的形状指标可单独列出。阈值应在开发数据上预先确定，不能为使测试通过而放宽。



## 8.3 必要对照与诊断



- 同数据、同种子和预算比较训练归一化/采样/latent 正则，以 val 几何和语义指标判断，不跨目标比较总 loss。

- 普通语义与加入 SemanticXY；不同初始化幅度及多初值；单视图与多视图；固定 prior 与合理范围内的 prior 权重。

- 完整存在牙与明确缺牙；有牙但遮挡不能替代缺牙样例；保留低匹配和 OT 不收敛状态。

- 局部固定曲面/材质对应有限差分与重新提取/光栅化后的实际下降分开记录，报告离散切换限制。

- 同模型固定均值牙龈与独立扫描牙龈差异分开诊断，必要时研究牙龈 nuisance 参数。



---



# 第九章 第一代系统结构与研究核心



## 9.1 第一代系统的整体结构



```text

上颌扫描 train ─→ 上颌 decoder / 训练码统计 ┐

下颌扫描 train ─→ 下颌 decoder / 训练码统计 ┤

原始 val ─→ 冻结 decoder、临时拟合 latent ─┤ 选模与 bundle

                                           ↓

已知相机 + 静态多视图逐牙语义 + 明确 presence

                                           ↓

均值候选初始化 → 两颌整体位姿和耦合 latent

                                           ↓

混合隐式表面 / 局部梯度桥 → 共同深度渲染 → Semantic + XY

                                           ↓

SemanticXY / dense / prior → 分阶段拟合与实际下降检查

                                           ↓

世界 mm 网格 + 拟合诊断 → 独立 truth 评价

```



## 9.2 整个项目最核心的研究逻辑



网络通过三维扫描学习可表达的牙列空间；照片掩码通过已知投影约束选择这个空间中的实例。先验训练和病例图像拟合是不同优化问题，decoder 不在照片拟合中更新。两个单颌先验可以共同解释静态双颌照片，而不需要先学习一个跨颌生成模型。



---



# 第十章 后续需要进一步明确的问题



## 10.1 已确定的接口与仍需实验回答的问题



### 10.1.1 先验模型的具体接口



本版确定使用 neural implicit DMM 和耦合组件表示，不再将 PCA/MBTR/其他先验同时列为当前未决选择。模型查询、单位、FDI、维数、presence 和统计来源均绑定 bundle。



### 10.1.2 latent 的统计与有效范围



使用原始训练存在牙代码的正则化协方差和 Cholesky 因子，拟合在 q 空间进行。数值可逆不证明概率校准；需要解码均值及不同幅度扰动检查几何，验证集决定有效范围和 prior 强度。不得用 val/test 代码重估训练 prior。



### 10.1.3 何时改为形态与排列解耦模型



当前先完成耦合 DMM 的可靠训练。少量训练病例也拟合不好时先查配方；训练好而验证差时查泛化与容量；三维拟合好而图像差时查初始化/渲染/优化。只有排列变化持续依赖不合理牙冠变形代偿时，再设计逐牙局部坐标、刚体和形态解耦的新主版本。



### 10.1.4 相机参数怎么获取



第一阶段使用合成器已知相机，不开展相机恢复。真实照片阶段另行研究标定、SfM/脸部估计、尺度和软组织遮挡；先验拟合不得通过自由调整相机掩盖几何错误。



### 10.1.5 SemanticXY 的当前选择与待验证项



当前已有 soft semantic、归一化 XY、OT/dustbin、采样质量、有效区域和梯度路径实现。仍需用真实牙齿表面验证采样量、语义/空间权重、未匹配代价、局部/全渲染更新及初始化范围；不是重新从零编写匹配器。



---



# 第十一章 当前阶段工程优先级



## 11.1 正式长训前的执行顺序



1. 统一双颌目标、10 维当前 specs、mm、presence 和显式像素约定；保持历史实验边界。

2. 明确训练配方版本、有效点分母、各项权重、采样、分组学习率和整轮日志。

3. 接入冻结 decoder 的 val-latent 拟合、独立评价、失败保留和 best_val；测试集封存。

4. 完整训练快照与 decoder 兼容契约分开，验证严格续跑、导出和新拟合器兼容。

5. 实现均值候选初始化并验证局部梯度、共同遮挡、回溯和网格刷新后的下降。

6. 用有限病例过拟合、短程独立验证和资源测量选择配方；源码检查不能替代这一步。

7. 冻结所需源码与配置后，分别正式训练上、下颌；不预先承诺固定 epoch 数足够。

8. 对训练模型做几何/统计验收，再做同模型合成和保留扫描合成重建；按证据决定模型结构后续修改。



当前源码新增能力不等于第六至八步已经完成。尚无依据把现有夹具、历史上颌 sanity 或低图像误差称为新双颌模型质量验收。



---



# 第十二章 核心理解总结



## 12.1 当前阶段最重要的理解总结



### SemanticXY 解决的是：





$$

\boxed{

\text{“模型离目标很远时，怎么知道应该往哪里移动？”}

}

$$



### 多视图解决的是：





$$

\boxed{

\text{“单张照片的 2D→3D 歧义如何减少？”}

}

$$



### Dental Prior 解决的是：





$$

\boxed{

\text{“3D 牙齿可以沿哪些合理方向变化？”}

}

$$



### Prior Loss 解决的是：





$$

\boxed{

\text{“在这些变化方向中，哪些参数位置更符合真实人群分布？”}

}

$$



最终：





$$

\boxed{

\text{2D evidence}

+

\text{geometry prior}

+

\text{statistical prior}

\rightarrow

\text{personalized 3D reconstruction}

}

$$



---



## 12.2 一句话概括我们的路线



> **利用多张牙齿照片提供的逐牙二维观测，通过 SemanticXY 建立稳定的长距离 2D–3D 对齐，在 Dental Morphable / Statistical Prior 的约束下联合优化牙齿形状、排列与整体位姿，最终恢复符合图像证据且统计上合理的个体化三维牙列。**

