# 多视图牙齿照片驱动的先验模型 3D 重建：从 SemanticXY 到 Prior Regularization

> 基于论文 **《Teeth Reconstruction and Performance Capture Using a Phone Camera》** 的第 4.2 节，以及围绕“多张牙齿照片 → DMM/牙齿先验 → 3D牙齿重建”的讨论整理。
>
> 本文不是对原论文逐字复述，而是将论文中的关键机制与我们计划中的静态多视图牙齿重建任务重新组织成一条更清晰的技术路线。

---

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

第一阶段暂时去掉 FLAME、动态 jaw tracking、表情参数和复杂遮挡策略，只保留：

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

所以完整系统本质上在寻找：


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

DMM 的 3D mesh 顶点同样带有每颗牙的 semantic label。

然后通过可微渲染器 $\hat S=\Pi(M,S)$，得到模型在当前相机下的 semantic mask。

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

因此优化器获得：

$$
\Delta x=200
$$

这样的长距离对应关系。

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

其中：

$$
SU=[S,U]
$$

包含：

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

两者共同作用。

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

通常 $z=0$ 附近对应训练数据分布的中心或平均状态。

如果 $z=[0.2,-0.3,0.1,\ldots]$，通常只是轻微偏离。

而如果优化变成 $z=[12,-18,25,\ldots]$，说明模型可能已经跑到了训练分布很远的位置。

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

则：$z\sim N(0,I)$。

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

如果 latent 已经 whitened $z\sim N(0,I)$，那么 $\|z\|^2$，就是合理 prior。

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

训练时网络通常只见过 latent space 中某些区域，例如：$z\approx[-2,2]^d$。

但优化器如果不受限制，完全可能走到：$z=[8,-12,15,\ldots]$。

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

对应 $L_{prior}$。

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

论文定义类似：


$$
L_{contact}
=
\|
\max(-\Phi(\cdot),0)
\|_1
$$

如果 $\Phi>0$，表示点位于牙齿外 $L=0$，如果 $\Phi=-2$，则 $-\Phi=2$，产生惩罚。

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

如果某张照片只看得到 $11,12,13,21,22,23$，那么这一帧就只监督这些可见牙齿 $m_{vis}=6$，看不到的牙齿：


$$
\boxed{\text{不监督}}
$$

而不是：


$$
\boxed{\text{认为它不存在}}
$$

然后依靠：

- 其他视角；
- dental prior；

决定这些不可见区域。

---

# 第六章 静态多视图牙齿重建方案

## 6.1 系统输入与输出

目标：

$$
\boxed{\text{同一个人的多张牙齿照片} \rightarrow \text{个体化3D牙列}}
$$

输入：

$$
\mathcal I = \{I_1,I_2,\dots,I_N\}
$$

每张照片：

$$
(I_k,C_k,S_k)
$$

其中：

- $I_k$：RGB 图像；
- $C_k$：相机参数；
- $S_k$：逐牙 semantic mask。

输出：

$$
z^*
$$

以及牙列整体刚性变换：

$$
M_{up},M_{low}
$$

最终得到：

$$
T(z^*)
$$

---

## 6.2 第一代系统的参数设计

原论文同时优化：

$$
\beta,\theta,\psi,z,M_{up},M_{low}
$$

但是静态牙齿任务不需要：

- face identity；
- expression；
- jaw motion。

因此第一版只保留：

$$
\boxed{\Theta= \{z,M_{up},M_{low}\}}
$$

其中：

| 参数 | 含义 | 是否跨视角共享 |
| --- | --- | --- |
| $z$ | 牙齿 shape latent | 是 |
| $M_{up}$ | 上颌整体刚性变换 | 是 |
| $M_{low}$ | 下颌整体刚性变换 | 是 |

相机：

$$
C_k
$$

第一阶段固定。

原因：

shape、pose、camera 存在严重 ambiguity。

例如：

牙齿变窄可能来自：

1. 真实牙冠较窄；
2. 牙列旋转；
3. 相机视角变化。

因此：

$$
\boxed{\text{先固定相机，再优化牙齿}}
$$

---

## 6.3 Representation Adaptation：DMM 到可微渲染

DMM 是 neural SDF-based dental prior。

因此：

$$
z \rightarrow \Phi(x;z)
$$

得到隐式牙齿表面。

但是 differentiable renderer 通常需要 mesh。

因此需要：

$$
\Phi(x;z) \rightarrow T(z)
$$

即：

$$
SDF \rightarrow Mesh
$$

随后：

$$
T(z) \rightarrow Renderer
$$

计算：

$$
L_{image}
$$

反向时：

$$
\frac{\partial L}{\partial M}
$$

需要通过 representation adaptation 回传：

$$
M \rightarrow \Phi \rightarrow z
$$

因此完整优化链：

$$
\boxed{z \rightarrow DMM \rightarrow SDF \rightarrow Mesh \rightarrow Renderer \rightarrow L \rightarrow z}
$$

---

## 6.4 采集条件限制

第一阶段：

> 被试保持同一个露齿状态，只移动相机。

原因：

如果不同照片：

$$
M_{low}^{(1)} \neq M_{low}^{(2)}
$$

问题会变成：

$$
\text{多视图重建} + \text{jaw tracking}
$$

复杂度明显增加。

因此：

第一版假设：

$$
\boxed{\text{静态牙列}}
$$

---

## 6.5 分阶段优化策略

### M0：Forward Validation

首先验证：

$$
z \rightarrow T(z) \rightarrow Renderer
$$

确保：

- DMM 可以生成 mesh；
- semantic attribute 正确；
- camera projection 正确。

---

### M1：Pose-only Optimization

目的：

验证 SemanticXY 是否能够恢复牙列位置。

固定：

$$
z=z_{GT}
$$

只优化：

$$
M_{up},M_{low}
$$

Loss：

$$
L = \lambda_{semXY}L_{semXY} + \lambda_{sem}L_{sem}
$$

该阶段回答：

> 给定正确牙齿形状，SemanticXY 是否可以找到正确 pose？

---

### M2：Shape-only Optimization

固定：

$$
M=M_{GT}
$$

优化：

$$
z
$$

目标：

验证：

$$
\frac{\partial L}{\partial z}
$$

是否正确。

如果失败：

问题通常来自：

- representation adaptation；
- latent optimization；
- gradient path。

---

### M3：Joint Shape + Pose Optimization

开放：

$$
\Theta = \{z,M_{up},M_{low}\}
$$

优化：

$$
z,M
$$

目标：

模拟真实重建。

---

### M4：加入 Prior Regularization

最终 loss：

$$
\boxed{L = \sum_k \left( \lambda_{semXY}L_{semXY}^{k} + \lambda_{sem}L_{sem}^{k} \right) + \lambda_zL_{prior}}
$$

其中：

$$
L_{prior} = \|z\|_2^2
$$

作用：

限制 latent 不偏离训练分布。

---

## 6.6 后续扩展

稳定后再加入：

- camera refinement；
- tooth pose prior；
- contour loss；
- RGB loss；
- gingival constraint；
- occlusion handling。

第一代系统不加入：

- FLAME；
- expression；
- contact loss；
- dynamic occlusion strategy。

因为这些属于动态 performance capture 问题。

---

# 第七章 工程实施路线

## 7.1 M0：多视图数据标准化

建议先准备 $8\sim15$，张同一人的牙齿照片。

覆盖：

- 正面；
- 左右斜侧；
- 略上；
- 略下；
- 可能的话增加更侧面的视角。

需要：

- 相机内参；
- 相机外参；
- 统一尺度或合理 scale convention；
- 原始完整图；
- 牙齿 ROI。

相机位姿最好提前通过：

- MetaShape；
- COLMAP；
- 标定板；
- 或其他 SfM/标定方式；

得到。

第一版不要同时让牙齿模型和相机一起完全自由优化。

---

### 7.1.1 为什么相机是最大的歧义来源之一

假设照片里某颗牙看起来比较窄。

原因可能是：

1. 牙齿本身较窄；
2. 牙列发生旋转；
3. 相机视角不同；
4. 相机内参估计有误。

所以如果同时自由优化 $z$，和 $C_k$，就会产生非常大的 shape-camera ambiguity。

因此建议：


$$
\boxed{
先固定相机，后重建牙齿
}
$$

流程稳定以后再允许：


$$
C_k
=
C_k^0+\Delta C_k
$$

做小范围 camera refinement。

---

### 7.1.2 为什么最好保留整张脸

即使最后只优化牙齿，也建议采集时保留脸部区域。

因为：

```text
整张脸
↓
更适合 SfM / Camera Pose Estimation

牙齿 ROI
↓
用于最终牙齿 Semantic / Geometry Optimization
```

仅靠一个牙齿 crop：

- 纹理重复；
- 高光明显；
- 局部特征有限；

往往并不利于相机估计。

---

## 7.2 M1：逐牙 Semantic Supervision

每张照片构建：


$$
S_k(x,y)
=
\text{tooth ID}
$$

理想情况下不是：

```text
tooth / background
```

而是：

```text
11 / 12 / 13 / 21 / 22 / 23 / ...
```

因为 SemanticXY 真正发挥价值的前提之一就是：

> 优化器知道“哪一颗模型牙”应该对应“哪一颗真实牙”。

早期实验完全可以接受：

- 人工 mask；
- SAM 辅助；
- 手工校正。

先验证几何优化闭环。

自动逐牙分割可以后置。

---

## 7.3 M2：冻结 Shape，只优化整体刚性位姿

建议这是第一个真正的优化实验。

固定 $z=z_{mean}$，只优化 $M_{up},M_{low}$，甚至第一版可以进一步先只用一个统一 $M_{arch}$，主要 loss：


$$
L_{semXY}
+
L_{sem}
$$

目的：

> 先让平均牙列坐到图像中正确的位置。

这个阶段非常适合验证 SemanticXY 是否真的发挥作用。

---

## 7.4 M3：开放牙齿先验参数

当整体位置稳定后，再开放 $z$，优化 $\{M,z\}$，一次迭代大致：
$$
z \rightarrow T(z) \rightarrow \Pi(T(z), C_k) \rightarrow (\hat{S}_k,\hat{U}_k) \rightarrow L \rightarrow \frac{\partial L}{\partial z}
$$
关键是：


$$
\boxed{
所有视角共享同一个 z
}
$$

所以不同照片会共同约束同一个 3D 牙列。

---

### 7.4.1 多视图的真正意义

例如：

- 正面照片对门牙宽度、排列提供强约束；
- 侧面照片对前突、深度方向提供信息；
- 左右斜侧能够约束犬牙、前磨牙及牙弓曲率；
- 上下角度可以补充牙冠高度与部分咬合关系。

所以每张照片只看到部分信息。

但共享 $z$，会迫使所有视角共同解释同一个 3D 结构。

---

## 7.5 M4：加入 Prior Regularization

此时第一版核心 loss 可以写成：


$$
\boxed{
L
=
\lambda_{semXY}L_{semXY}
+
\lambda_{sem}L_{sem}
+
\lambda_{prior}L_{prior}
}
$$

多视图：


$$
L
=
\sum_{k=1}^{N}
\left[
\lambda_{semXY}
L_{semXY}^{(k)}
+
\lambda_{sem}
L_{sem}^{(k)}
\right]
+
\lambda_{prior}
L_{prior}
$$

其中 $z,M_{up},M_{low}$，是全局共享变量。

而 $C_k$，是每个视角单独的相机参数。

---

## 7.6 M5：必要时开放每颗牙的 Pose

如果发现单纯 shape prior 无法表达真实牙齿排列，可以引入每颗牙的局部刚体变换 $P_j=(R_j,t_j)$，但不建议直接开放 $28\times6DoF$，因为这样每颗牙都可能自由乱跑。

更合理的是使用 pose prior：


$$
p
=
\bar p
+
B_{pose}\alpha
$$

其中 $\alpha$，是低维 pose coefficient。

这样最终照片优化的是 $\alpha$；而不是几十颗牙完全独立的 pose。

这与我们的 MBTR 式 shape prior / pose prior 思路非常吻合。

---

## 7.7 M6：加入更复杂的图像信息

第一版不建议立即加 RGB photometric loss。

原因：

- 牙齿高反光；
- 曝光变化明显；
- 颜色差异小；
- 光照模型不好时反而容易误导几何。

建议顺序：

第一阶段：


$$
SemanticXY
+
Semantic
+
Prior
$$

跑通以后再逐步加入：

- tooth contour；
- silhouette；
- landmark；
- gingival margin；
- RGB；
- texture；
- SDF/contact；
- 更复杂的 occlusion constraint。

---

# 第八章 验证策略与消融实验

## 8.1 Synthetic Sanity Check

真人照片之前，建议先做合成实验。

从我们的先验中采样 $z_{GT}$，生成 $T(z_{GT})$，设置多个虚拟相机 $C_1,\ldots,C_8$，渲染 semantic masks $S_1,\ldots,S_8$，然后故意从 $z=0$，以及错误位姿 $M=M_{wrong}$，开始优化。

观察：


$$
z^*
\stackrel{?}{\approx}
z_{GT}
$$

---

## 8.2 Synthetic 实验重点检查什么

如果 synthetic 都恢复不了，问题大概率存在于：

- renderer；
- semantic encoding；
- SemanticXY；
- camera convention；
- gradient；
- coordinate system；
- optimization schedule；
- learning rate；
- pose parameterization。

而不是：

> “真实照片太复杂”。

---

## 8.3 SemanticXY Ablation

故意给初始化 $\Delta x,\Delta y,\Delta z,\Delta R$，一个比较大的偏差。

分别跑 $\text{with SemanticXY}$，与 $\text{without SemanticXY}$，比较：

- 收敛率；
- 最终 mask IoU；
- 最终 pose error；
- 最终 shape parameter error；
- 是否卡 local minimum。

这个实验可以作为我们项目最早期的关键验证之一。

---

# 第九章 第一代系统结构与研究核心

## 9.1 第一代系统的整体结构

```text
                 多张 RGB 照片
                      │
        ┌─────────────┴─────────────┐
        │                           │
  Camera Calibration           逐牙分割
        │                           │
      K,R,t               per-tooth semantic
        │                           │
        └─────────────┬─────────────┘
                      │
                      ▼
                Dental Prior
                     z
                      │
                      ▼
                 3D Teeth
                      │
               Mup / Mlow
                      │
                      ▼
          Differentiable Renderer
             ↙                ↘
       Semantic               XY
          │                    │
          └────────┬───────────┘
                   ▼
              SemanticXY
                   │
          ┌────────┴────────┐
          │                 │
       LsemXY             Lsem
          │                 │
          └────────┬────────┘
                   +
                Lprior
                   │
                   ▼
             Backpropagate
                   │
           update z / M
                   │
                   ▼
        Personalized 3D Teeth
```

---

## 9.2 整个项目最核心的研究逻辑

可以浓缩成：


$$
\boxed{
\text{Multi-view Image Evidence}
+
\text{Dental Shape/Pose Prior}
+
\text{SemanticXY Correspondence}
}
$$

目标不是：

> 训练一个神经网络，从照片直接猜 3D。

而是：

> 在已有牙齿先验空间中，通过多视图二维证据寻找最符合该个体的 3D 实例。

也就是说：


$$
\boxed{
\text{照片不是负责创造牙齿，
而是负责告诉先验模型应该走到哪里。}
}
$$

---

# 第十章 后续需要进一步明确的问题

## 10.1 当前最值得继续深入的问题

接下来的核心问题已经比较明确。

### 10.1.1 先验模型的具体接口

需要明确 $T(z)$，到底来自：

- neural SDF DMM；
- PCA shape model；
- MBTR 式 shape prior；
- shape + pose 双先验；
- 还是我们自己的统计模型。

---

### 10.1.2 latent 是否标准化

必须搞清楚 $z$，究竟是：

- 原始 PCA coefficient；
- whitened PCA coefficient；
- neural latent；
- auto-decoder embedding。

因为这决定 $L_{prior}$，到底应该写成 $\|z\|^2$，还是 $z^T\Sigma^{-1}z$，或其他概率形式。

---

### 10.1.3 shape 与 pose 是否分离

最好区分 $z_{shape}$，与 $z_{pose}$，因为：

- tooth morphology；
- dental arch arrangement；
- individual tooth pose；

是不同层面的变化。

对于我们已经在复现的 MBTR 式统计牙齿模型，这一点尤其重要。

---

### 10.1.4 相机参数怎么获取

需要单独验证：

- MetaShape；
- COLMAP；
- 手机标定；
- SfM；
- face-based pose estimation；

哪一种更适合我们的采集环境。

---

### 10.1.5 SemanticXY 如何工程实现

需要明确：

- semantic encoding；
- XY normalization；
- OT cost；
- semantic weight；
- spatial weight；
- Sinkhorn / transport 求解；
- visibility mask；
- 不同牙齿类别之间是否允许 transport；
- 背景如何处理；
- gradient 如何回到 vertex / latent。

这部分应当单独做一个最小实验，而不是直接塞进完整系统。

---

# 第十一章 当前阶段工程优先级

## 11.1 推荐推进顺序

建议按照以下顺序推进：

```text
1. Dental Prior 能生成 Mesh
        ↓
2. Mesh 可以被可微渲染
        ↓
3. Render per-tooth Semantic
        ↓
4. 单视角普通 Semantic Loss 可优化 Pose
        ↓
5. 加 SemanticXY
        ↓
6. 做大位姿偏差 Synthetic Test
        ↓
7. 加多视图共享参数
        ↓
8. 开放 Shape Latent
        ↓
9. 加 Prior Loss
        ↓
10. 用真实照片验证
        ↓
11. 再逐步加 Pose Prior / Landmark / RGB / Gingiva
```

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