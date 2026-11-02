'use strict';
// 按授课顺序补充符号定义与推导；原图解和交互页面保持原位。
(() => {
  const slides=window.LECTURE,F=window.LectureFigures;
  const raw=String.raw;
  const math=(tex,inline=false)=>`<${inline?'span':'div'} class="${inline?'math-inline':'math'}" data-tex="${tex.replaceAll('&','&amp;').replaceAll('"','&quot;')}"></${inline?'span':'div'}>`;
  const eq=tex=>math(tex);
  const mi=tex=>math(tex,true);
  const points=(...items)=>`<ul class="points">${items.map(x=>`<li>${x}</li>`).join('')}</ul>`;
  const symbols=rows=>`<dl class="symbols">${rows.map(([s,d])=>`<div><dt>${mi(s)}</dt><dd>${d}</dd></div>`).join('')}</dl>`;
  const fig=(body,caption)=>`<figure>${body}<figcaption>${caption}</figcaption></figure>`;
  const split=(left,right)=>`<div class="lecture-split"><div>${left}</div><div>${right}</div></div>`;
  const lookup=title=>{const s=slides.find(s=>s.title===title);if(!s)throw Error(`缺少课件页：${title}`);return s;};
  function replace(title,body,cls=''){const s=lookup(title);s.body=body;s.cls=cls;}
  function before(title,items){const i=slides.indexOf(lookup(title));slides.splice(i,0,...items.map(s=>({...s,part:s.part||slides[i].part,notes:s.notes||'推导在本页列出的假设下成立；符号、单位和中间步骤均在正文中给出。'})));}
  function after(title,items){const i=slides.indexOf(lookup(title));slides.splice(i+1,0,...items.map(s=>({...s,part:s.part||slides[i].part,notes:s.notes||'推导在本页列出的假设下成立；符号、单位和中间步骤均在正文中给出。'})));}

  replace('课程路线与符号约定',split(`<h3>四个问题</h3>${points('晶体如何由点阵与结构基元构成？','为什么特定方向出现衍射？','如何由像素估计强度与结构？','agent 如何组织计算和证据？')}`,`<h3>阅读数学表达式</h3>${symbols([[raw`\mathbf v`,'粗体小写：列向量；非粗体小写通常为标量。'],[raw`M^{\mathsf T},\ M^{-1}`,'矩阵转置、逆矩阵；逆要求矩阵可逆。'],[raw`\|\mathbf v\|,\ |z|`,'向量的 Euclidean 范数、复数的模。'],[raw`\mathbb Z,\ \mathbb R,\ \mathbb C`,'整数、实数、复数集合。']])}<p class="footnote">全课长度以 Å、倒易长度以 Å⁻¹ 表示。新符号在首次使用前或同页定义；含义改变时重新定义。</p>`));

  before('晶体 = 点阵 + 结构基元',[
    {title:'点阵表示：先定义坐标与基矩阵',body:split(symbols([[raw`\mathbf a,\mathbf b,\mathbf c`,'三条线性无关的直接基矢，单位 Å；本页选用原胞点阵基。'],[raw`B=[\mathbf a\ \mathbf b\ \mathbf c]`,'3×3 直接晶胞矩阵；三条基矢按列排列，单位 Å。'],[raw`\mathbf n=(n_1,n_2,n_3)^{\mathsf T}`,'整数平移标签，各分量属于 ℤ、无量纲。'],[raw`\mathbf R_{\mathbf n}`,'标签 n 所对应点阵点的笛卡尔位置，单位 Å。']]),`${eq(raw`\mathbf R_{\mathbf n}=n_1\mathbf a+n_2\mathbf b+n_3\mathbf c=B\mathbf n`)}${fig(F.motif(),'图 1　原胞点阵位置重复同一结构基元；惯用中心化胞还需加入中心化平移。')}`)},
  ]);
  replace('晶体 = 点阵 + 结构基元',split(`${symbols([[raw`j`,'基元中原子的编号。'],[raw`\mathbf x_j=(x_j,y_j,z_j)^{\mathsf T}`,'第 j 个原子的分数坐标，无量纲。'],[raw`\mathbf r_{\mathbf n j}`,'平移 n 后，第 j 个原子的笛卡尔位置，Å。']])}${eq(raw`\mathbf r_{\mathbf n j}=B\mathbf x_j+B\mathbf n=B(\mathbf x_j+\mathbf n)`)}${points('点阵描述平移；结构基元描述每个平移位置上的内容。','点阵点是数学位置，不必恰好是原子。')}`,fig(F.motif(),'图 1　晶体由点阵与同一结构基元组成。')));
  replace('晶胞、基矢与分数坐标',split(`${symbols([[raw`\mathbf x=(x,y,z)^{\mathsf T}`,'无量纲分数坐标；r 为对应笛卡尔位置（Å）。'],[raw`a,b,c`,'基矢 a、b、c 的长度，Å。'],[raw`\alpha,\beta,\gamma`,'分别为 b–c、a–c、a–b 的夹角。']])}${eq(raw`\mathbf r=x\mathbf a+y\mathbf b+z\mathbf c=B\mathbf x`)}${eq(raw`a=\|\mathbf a\|,\quad b=\|\mathbf b\|,\quad c=\|\mathbf c\|`)}`,fig(F.cell(),'图 2　晶胞投影；屏幕上的投影角不等于真实轴角。')));
  after('晶胞、基矢与分数坐标',[
    {title:'晶胞度量：把分数坐标变成真实距离',body:`${symbols([[raw`\Delta\mathbf x,\ \Delta\mathbf r`,'两点的分数位移（无量纲）与笛卡尔位移（Å）。'],[raw`G=B^{\mathsf T}B`,'3×3 直接度量矩阵，单位 Å²。'],[raw`V_{\mathrm{cell}}=|\det B|`,'晶胞体积，单位 Å³；下标 cell 避免与噪声方差混淆。']])}${eq(raw`\|\Delta\mathbf r\|^2=(B\Delta\mathbf x)^{\mathsf T}(B\Delta\mathbf x)=\Delta\mathbf x^{\mathsf T}G\Delta\mathbf x`)}${eq(raw`G=\begin{pmatrix}a^2&ab\cos\gamma&ac\cos\beta\\ab\cos\gamma&b^2&bc\cos\alpha\\ac\cos\beta&bc\cos\alpha&c^2\end{pmatrix}`)}`,cls:'derivation'},
  ]);
  lookup('原胞与惯用胞').body=lookup('原胞与惯用胞').body.replace('<div class="lecture-split">','<p class="definition-line">N<sub>P</sub>、N<sub>I</sub>、N<sub>F</sub>：对应 P、I、F 惯用胞中按共享比例计数的点阵点数，均无量纲。</p><div class="lecture-split">');
  lookup('七晶系：惯用晶胞与特征对称性').body+='<p class="footnote">对称符号 n 表示 n 重旋转，−n 表示旋转反演；m 表示镜面。2/m 表示二重轴与垂直镜面的组合，不是数值比值。</p>';
  lookup('十四种 Bravais 点阵').body+='<p class="footnote">点阵符号首字母 a、m、o、t、h、c 依次指三斜、单斜、正交、四方、六方晶族、立方；后缀为中心化类型。此处 a、c 是类型代码，不是轴长。</p>';
  before('点群、空间群与等价位置',[
    {title:'对称操作中的矩阵与平移',body:`${symbols([[raw`R`,'作用于分数坐标的 3×3 点操作矩阵，无量纲；可为旋转、反演等。'],[raw`\mathbf t`,'对称操作的分数平移向量，无量纲，可含半整数等分量。'],[raw`\mathbf x'`,'变换后的分数坐标；模 ℤ³ 表示整数晶胞平移等价。']])}${eq(raw`\mathbf x'=R\mathbf x+\mathbf t\pmod{\mathbb Z^3}`)}${eq(raw`\|BR\Delta\mathbf x\|^2=\Delta\mathbf x^{\mathsf T}R^{\mathsf T}GR\Delta\mathbf x`)}${points('对所有位移都保持真实距离，要求 RᵀGR=G。','平移 t 在两点作差时抵消；不能从强度相等推断其相位也相等。')}`},
  ]);
  before('晶向、晶面与 Miller 指数',[
    {title:'Miller 指数：由截距得到整数标签',body:split(`${symbols([[raw`(p_a a,p_b b,p_c c)`,'晶面与三轴的截距；pₐ、pᵦ、p𝚌 是无量纲截距倍数。'],[raw`h,k,l`,'截距倍数的倒数经通分得到的整数，定义晶面取向。'],[raw`n\in\mathbb Z`,'同一平行晶面族中不同平面的编号。']])}${eq(raw`\frac{x}{p_a}+\frac{y}{p_b}+\frac{z}{p_c}=1`)}${eq(raw`(p_a,p_b,p_c)=(1,1,\tfrac12)\Rightarrow(hkl)=(112)`)}`,fig(F.miller(),'图 1　截距 a、b、c/2；倒数为 1、1、2。'))},
  ]);
  lookup('晶向、晶面与 Miller 指数').body=lookup('晶向、晶面与 Miller 指数').body.replace('<div class="lecture-split">','<p class="definition-line">[uvw] 中 u、v、w 是无量纲整数方向指标；∥ 表示平行。晶面方程中的 x、y、z 是分数坐标，n 为平面编号。</p><div class="lecture-split">');
  before('倒易基矢与晶面间距',[
    {title:'倒易基矢的定义与构造',body:`${symbols([[raw`\mathbf a^*,\mathbf b^*,\mathbf c^*`,'直接基矢的对偶基矢，单位 Å⁻¹；星号表示倒易，不表示复共轭。'],[raw`A=[\mathbf a^*\ \mathbf b^*\ \mathbf c^*]`,'3×3 倒易基矩阵，单位 Å⁻¹。'],[raw`I_3`,'3×3 单位矩阵；与反射强度 I 不同。']])}${eq(raw`B^{\mathsf T}A=I_3\quad\Rightarrow\quad A=B^{-\mathsf T}`)}${eq(raw`\mathbf a^*=\frac{\mathbf b\times\mathbf c}{\mathbf a\cdot(\mathbf b\times\mathbf c)}`)}${points('× 表示向量叉积；分母是有符号体积，要求基矢线性无关。','另外两条倒易基矢由循环置换得到，不额外乘 2π。')}`},
  ]);
  lookup('倒易基矢与晶面间距').body=lookup('倒易基矢与晶面间距').body.replace('B^{\\mathsf T}A=I','B^{\\mathsf T}A=I_3');
  replace('倒易基矢与晶面间距',split(`${symbols([[raw`\mathbf h=(h,k,l)^{\mathsf T}`,'非零整数反射指标向量（h,k,l 不全为零），无量纲。'],[raw`\mathbf q_{hkl}=A\mathbf h`,'与 (hkl) 晶面垂直的倒易矢量，Å⁻¹。'],[raw`d_{hkl}`,'相邻 hx+ky+lz=n 晶面的法向间距，Å。']])}${eq(raw`\mathbf q_{hkl}=h\mathbf a^*+k\mathbf b^*+l\mathbf c^*`)}${eq(raw`d_{hkl}=\|\mathbf q_{hkl}\|^{-1}`)}`,fig(F.reciprocal(),'图 1　倒易基矢垂直相应直接晶面；图中显示方向关系。')));
  after('倒易基矢与晶面间距',[
    {title:'推导：晶面间距为什么等于倒易长度的倒数',body:`${eq(raw`\mathbf q_{hkl}^{\mathsf T}\mathbf r=(A\mathbf h)^{\mathsf T}B\mathbf x=\mathbf h^{\mathsf T}\mathbf x=n`)}${points('相邻晶面分别满足 qᵀr=n 与 qᵀr=n+1。','沿单位法向 q/‖q‖ 移动距离 d，代入两平面方程作差。')}${eq(raw`\mathbf q^{\mathsf T}\left(d\frac{\mathbf q}{\|\mathbf q\|}\right)=1\quad\Rightarrow\quad d\|\mathbf q\|=1`)}${eq(raw`d_{hkl}^{-2}=\mathbf h^{\mathsf T}A^{\mathsf T}A\mathbf h=\mathbf h^{\mathsf T}G^{-1}\mathbf h`)}`,cls:'derivation'},
    {title:'实例：正交晶胞与立方晶胞的面间距',body:`${eq(raw`G=\operatorname{diag}(a^2,b^2,c^2)\quad\text{（正交晶胞）}`)}${eq(raw`d_{hkl}^{-2}=\frac{h^2}{a^2}+\frac{k^2}{b^2}+\frac{l^2}{c^2}`)}${eq(raw`a=b=c\quad\Rightarrow\quad d_{hkl}=\frac{a}{\sqrt{h^2+k^2+l^2}}`)}${eq(raw`a=5\ \text{Å}:\quad d_{100}=5\ \text{Å},\quad d_{110}=\frac5{\sqrt2}\ \text{Å}`)}${points('diag 表示以所列元素为对角元的对角矩阵。','这是度量计算示例，不涉及具体材料或实测数据。')}`},
  ]);
  before('Bragg 定律：相邻晶面的光程差',[
    {title:'Bragg 构造中的符号与角度',body:split(symbols([[raw`\lambda`,'入射单色 X 射线波长，Å。'],[raw`\theta,\ 2\theta`,'掠射角、散射角；公式中的三角函数以一致角度单位计算。'],[raw`\Delta`,'两束散射路径之间的附加光程，Å。'],[raw`n_B=1,2,\ldots`,'Bragg 级次，无量纲；与晶面编号 n 区分。']]),`${eq(raw`\Delta=2d_{hkl}\sin\theta`)}${eq(raw`\Delta=n_B\lambda`)}${points('以下几何图中 A、B 是两晶面上的散射点标签，不是矩阵。','先作等相位入射波前，再比较相同出射方向上的相位。')}`)},
  ]);
  lookup('Bragg 定律：相邻晶面的光程差').body=lookup('Bragg 定律：相邻晶面的光程差').body.replaceAll('n\\lambda','n_B\\lambda').replace('写成 n=1','写成 n_B=1');
  after('Bragg 定律：相邻晶面的光程差',[
    {title:'推导：由光程差得到相长干涉',body:`${symbols([[raw`\delta\varphi`,'相邻晶面散射波的相位差，弧度；不是扫描角 φ。'],[raw`i`,'虚数单位，满足 i²=−1。']])}${eq(raw`\delta\varphi=\frac{2\pi}{\lambda}\Delta=\frac{4\pi d_{hkl}\sin\theta}{\lambda}`)}${eq(raw`1+e^{i\delta\varphi}\quad\text{在}\quad\delta\varphi=2\pi n_B\quad\text{时同相}`)}${eq(raw`\therefore\quad 2d_{hkl}\sin\theta=n_B\lambda`)}${points('把级次吸收为反射 (n_Bh,n_Bk,n_Bl)，便可统一使用一级条件。','动画中调节角度只改变光程差，不保证每个角度都相长。')}`,cls:'derivation'},
  ]);
  lookup('Bragg 定律与实验分辨率').body='<p class="definition-line">q 为出射与入射空间频率向量之差（Å⁻¹），在衍射位置等于 q<sub>hkl</sub>；d<sub>min</sub> 为最小可测晶面间距；θ<sub>max</sub> 为最大 Bragg 角，均指有效采集范围。</p>'+lookup('Bragg 定律与实验分辨率').body;
  before('弹性散射与 Laue 条件',[
    {title:'散射波矢：单位、方向与相位约定',body:`${symbols([[raw`\mathbf s_0,\mathbf s_1`,'入射、出射空间频率向量，均为 Å⁻¹；不含 2π。'],[raw`\mathbf q=\mathbf s_1-\mathbf s_0`,'散射矢量，Å⁻¹；在衍射位置等于 A h。'],[raw`\delta\varphi_{\mathbf n}`,'与原点相比，平移 Rₙ 产生的散射相位差。']])}${eq(raw`\|\mathbf s_0\|=\|\mathbf s_1\|=\lambda^{-1},\qquad \delta\varphi_{\mathbf n}=2\pi\mathbf q^{\mathsf T}B\mathbf n`)}${points('弹性散射保持波长，因此两波矢等长。','本课振幅采用正号指数；若改取相反散射矢量，反射指标和傅里叶符号也要一致转换。')}`},
    {title:'推导：晶格求和与 Laue 条件',body:`${symbols([[raw`N_a`,'沿 a 方向的晶胞数，正整数。'],[raw`\psi_a=2\pi\mathbf q\cdot\mathbf a`,'相邻 a 向晶胞的相位差，弧度。'],[raw`S_{N_a}`,'该方向 Nₐ 个晶胞散射的无量纲振幅和。']])}${eq(raw`S_{N_a}=\sum_{u=0}^{N_a-1}e^{iu\psi_a}=e^{i(N_a-1)\psi_a/2}\frac{\sin(N_a\psi_a/2)}{\sin(\psi_a/2)}`)}${points('几何级数在 ψₐ→2πh 时的极限为 Nₐ；有限晶体产生有限峰宽。','三方向同时相长给出 q·a=h、q·b=k、q·c=l。')}${eq(raw`B^{\mathsf T}\mathbf q=\mathbf h\quad\Rightarrow\quad\mathbf q=B^{-\mathsf T}\mathbf h=A\mathbf h`)}`,cls:'derivation'},
  ]);
  before('Ewald 球：倒易空间的衍射条件',[
    {title:'推导：Bragg 条件与 Ewald 方程等价',body:`${eq(raw`\|\mathbf s_0+\mathbf q\|^2=\|\mathbf s_0\|^2`)}${eq(raw`\|\mathbf s_0\|^2+2\mathbf s_0^{\mathsf T}\mathbf q+\|\mathbf q\|^2=\|\mathbf s_0\|^2`)}${eq(raw`2\mathbf s_0^{\mathsf T}\mathbf q+\|\mathbf q\|^2=0`)}${points('s₀ 与 s₁ 等长、夹角 2θ，向量差 q 是等腰三角形的底边。')}${eq(raw`\|\mathbf q\|=2\|\mathbf s_0\|\sin\theta=\frac{2\sin\theta}{\lambda}=\frac1{d_{hkl}}`)}<p class="footnote">O 为倒易原点、C 为 Ewald 球心、H 为反射节点，均为构图标签；H 与后文检测核权重无关。</p>`,cls:'derivation'},
  ]);
  lookup('旋转法：倒易节点如何穿过 Ewald 球').body='<p class="definition-line">φ：扫描旋转角；R(φ)：实验室坐标中的正交旋转矩阵；A：参考取向的倒易基；f(φ)：Ewald 残差，单位 Å⁻²，零点给出衍射角度。</p>'+lookup('旋转法：倒易节点如何穿过 Ewald 球').body;

  before('结构因子：位置相位与原子散射',[
    {title:'结构因子的符号与量纲',body:symbols([[raw`F_{\mathbf h}\in\mathbb C`,'反射 h 的晶胞结构因子；X 射线常用电子数作为振幅单位。'],[raw`o_j\in[0,1],\quad f_j(s)`,'原子 j 的物理占有率（无量纲）、原子散射因子（电子数）。'],[raw`s=\sin\theta/\lambda`,'散射变量，Å⁻¹；与向量 s₀、s₁ 不同。'],[raw`T_j,\quad U_j`,'热运动振幅因子（无量纲）、笛卡尔位移协方差矩阵（Å²）。'],[raw`I_{\mathbf h},\quad k_{\mathrm{sc}}`,'反射强度及正尺度因子；尺度把 |F|² 转为所用强度单位。']])},
    {title:'推导：电子密度的 Fourier 振幅',body:`${symbols([[raw`\rho_{\mathrm{atom},j}(\mathbf u)`,'以原子中心为原点的电子密度，电子数 Å⁻³。'],[raw`\mathbf u`,'相对原子中心的笛卡尔位移，Å。']])}${eq(raw`f_j(\mathbf q)=\int\rho_{\mathrm{atom},j}(\mathbf u)e^{2\pi i\mathbf q^{\mathsf T}\mathbf u}\,d^3\mathbf u`)}${eq(raw`\int\rho_{\mathrm{atom},j}(\mathbf r-\mathbf r_j)e^{2\pi i\mathbf q^{\mathsf T}\mathbf r}\,d^3\mathbf r=f_j(\mathbf q)e^{2\pi i\mathbf q^{\mathsf T}\mathbf r_j}`)}${points('平移密度只增加相位因子；运动学近似下各原子的振幅线性相加。','在衍射位置 q=A h、rⱼ=B xⱼ，因此 qᵀrⱼ=hᵀxⱼ。')}`,cls:'derivation'},
    {title:'推导：Gaussian 热运动的衰减因子',body:`${symbols([[raw`\mathbf u_j\sim\mathcal N(\mathbf0,U_j)`,'原子瞬时位移服从零均值三维 Gaussian；𝒩 表示正态分布。'],[raw`\mathbb E[\,\cdot\,]`,'对位移分布取期望，即时间/集合平均。'],[raw`U_{\mathrm{iso},j}`,'各向同性位移方差，Å²。']])}${eq(raw`T_j=\mathbb E[e^{2\pi i\mathbf q^{\mathsf T}\mathbf u_j}]=\exp\!\left[-\frac12(2\pi\mathbf q)^{\mathsf T}U_j(2\pi\mathbf q)\right]`)}${eq(raw`U_j=U_{\mathrm{iso},j}I_3\Rightarrow T_j=e^{-2\pi^2U_{\mathrm{iso},j}\|\mathbf q\|^2}=e^{-8\pi^2U_{\mathrm{iso},j}s^2}`)}${points('这是 Gaussian 随机位移的特征函数；衰减针对振幅，强度还需取模平方。')}`,cls:'derivation'},
  ]);
  lookup('结构因子：位置相位与原子散射').body=lookup('结构因子：位置相位与原子散射').body.replace('I_{hkl}\\propto |F_{hkl}|^2','I_{hkl}=k_{\\mathrm{sc}}|F_{hkl}|^2');
  before('系统消光：平移对称导致振幅相消',[
    {title:'推导：平移因子如何产生系统消光',body:`${symbols([[raw`\mathbf t_c=(\tfrac12,\tfrac12,\tfrac12)^{\mathsf T}`,'体心附加点阵点的分数平移。'],[raw`F^{(P)}_{hkl}`,'只对一个原始基元求和的结构因子；上标 P 为标签。']])}${eq(raw`F_{hkl}=\sum_j f_jT_jo_je^{2\pi i\mathbf h^{\mathsf T}\mathbf x_j}\left(1+e^{2\pi i\mathbf h^{\mathsf T}\mathbf t_c}\right)`)}${eq(raw`1+e^{\pi i(h+k+l)}=\begin{cases}2,&h+k+l\ \text{为偶数},\\0,&h+k+l\ \text{为奇数}.\end{cases}`)}${points('相消来自整套基元的平移副本，与单个元素的散射强弱无关。','21 螺旋沿 b 的半平移对 0k0 同样产生因子 1+exp(πik)。')}`,cls:'derivation'},
  ]);

  before('相位问题与电子密度重建',[
    {title:'Fourier 反演：模长、相位与电子密度',body:`${symbols([[raw`\varphi_{\mathbf h}`,'复结构因子 Fₕ 的相位角，弧度。'],[raw`\rho(\mathbf x)`,'分数位置 x 处的物理电子密度，电子数 Å⁻³。'],[raw`\Omega`,'实际观测反射指标的集合；完整 Fourier 展开则对全部整数指标求和。']])}${eq(raw`F_{\mathbf h}=V_{\mathrm{cell}}\int_{[0,1)^3}\rho(\mathbf x)e^{2\pi i\mathbf h^{\mathsf T}\mathbf x}\,d^3\mathbf x`)}${eq(raw`\rho(\mathbf x)=\frac1{V_{\mathrm{cell}}}\sum_{\mathbf h\in\mathbb Z^3}F_{\mathbf h}e^{-2\pi i\mathbf h^{\mathsf T}\mathbf x}`)}${points('Vcell 把分数体积转为真实体积；正反变换的符号相反。','有限反射集给出带限密度，缺失反射和相位会限制重建。')}`,cls:'derivation'},
  ]);
  lookup('相位问题与电子密度重建').body=lookup('相位问题与电子密度重建').body.replace('\\frac1V','\\frac1{V_{\\mathrm{cell}}}');
  after('相位问题与电子密度重建',[
    {title:'推导：为什么整体平移不改变强度',body:`${symbols([[raw`\mathbf o`,'对所有原子施加的同一分数平移，无量纲；不是原子占有率 oⱼ。'],[raw`F'_{\mathbf h}`,'整体平移后计算的结构因子。']])}${eq(raw`F'_{\mathbf h}=\sum_j o_jf_jT_je^{2\pi i\mathbf h^{\mathsf T}(\mathbf x_j+\mathbf o)}=e^{2\pi i\mathbf h^{\mathsf T}\mathbf o}F_{\mathbf h}`)}${eq(raw`|F'_{\mathbf h}|^2=|e^{2\pi i\mathbf h^{\mathsf T}\mathbf o}|^2|F_{\mathbf h}|^2=|F_{\mathbf h}|^2`)}${points('强度不能决定整体原点；比较恢复结构时应先对齐等价原点。','无异常散射时还可能有反演等价，不能直接按坐标表判失败。')}`,cls:'derivation'},
  ]);
  before('相位恢复：实空间与倒空间的交替约束',[
    {title:'幅度投影：保留相位，替换观测模长',body:`${symbols([[raw`\mathcal F,\ \mathcal F^{-1}`,'离散 Fourier 正、反变换算子；沿用本课符号约定。'],[raw`A_{\mathbf h}^{\mathrm{obs}}`,'由强度与尺度估计的观测幅度（电子数）；不是倒易基矩阵 A。'],[raw`P_{\mathcal M},\ P_{\mathcal A}`,'倒空间幅度约束算子、实空间原子性等约束算子。']])}${eq(raw`[P_{\mathcal M}(F)]_{\mathbf h}=A_{\mathbf h}^{\mathrm{obs}}\frac{F_{\mathbf h}}{|F_{\mathbf h}|},\quad \mathbf h\in\Omega,\ |F_{\mathbf h}|>0`)}${points('未观测反射通常保留当前估计；零振幅需选择明确的初相位约定。','这是无噪声/硬约束的教学表达；真实误差应由相应统计模型处理。')}`,cls:'derivation'},
  ]);
  lookup('相位恢复：实空间与倒空间的交替约束').body='<p class="definition-line">t 为迭代编号（非对称平移 t）；ρ⁽ᵗ⁾ 是第 t 步密度估计。下式按从右到左的顺序依次作用。</p>'+lookup('相位恢复：实空间与倒空间的交替约束').body.replaceAll('A_{\\mathbf h}','A_{\\mathbf h}^{\\mathrm{obs}}');
  before('结构精修：受约束的统计估计',[
    {title:'精修目标：观测、参数与限制的定义',body:symbols([[raw`\mathbf p,\quad\widehat{\mathbf p}`,'原子位置、位移和占有率等物理参数向量；帽号表示估计值。'],[raw`I_{\mathrm{obs},h},\ \sigma_h`,'反射 h 的实验强度及标准不确定度，单位相同；允许观测为负。'],[raw`k_{\mathrm{sc}}|F_h(\mathbf p)|^2`,'预测强度；ksc 是正尺度，使模型与观测单位一致。'],[raw`g_r(\mathbf p),\ g_{r,0},\ \tau_r`,'第 r 个限制的计算量、目标和容差；三者单位相同。'],[raw`\mathbf z,\quad\mathbf p(\mathbf z)`,'独立优化变量及满足严格约束的物理参数化。']])},
    {title:'推导：Gaussian 似然产生加权平方残差',body:`${symbols([[raw`\mu_h=k_{\mathrm{sc}}|F_h(\mathbf p)|^2`,'反射 h 的预测均值，与观测强度单位相同。'],[raw`L(\mathbf p),\ C`,'独立观测的联合似然；C 是与参数无关的常数。']])}${eq(raw`I_{\mathrm{obs},h}\sim\mathcal N(\mu_h,\sigma_h^2),\qquad L=\prod_h\frac{e^{-(I_{\mathrm{obs},h}-\mu_h)^2/(2\sigma_h^2)}}{\sqrt{2\pi}\sigma_h}`)}${eq(raw`-2\log L=\sum_h\frac{(I_{\mathrm{obs},h}-\mu_h)^2}{\sigma_h^2}+C`)}${points('固定 σₕ 时，最大化似然等价于最小化该平方残差和。','限制项是额外化学信息；严格约束则直接改变可行域。')}`,cls:'derivation'},
  ]);
  lookup('结构精修：受约束的统计估计').body=lookup('结构精修：受约束的统计估计').body.replace('k|F_h','k_{\\mathrm{sc}}|F_h');

  before('原始像素与自校准检测',[
    {title:'像素测量模型：变量与单位',body:symbols([[raw`p,\ a`,'有效像素/帧体素编号、贡献反射编号；p 在本节不是结构参数向量。'],[raw`N_p,\quad b_p`,'光子等效计数及其背景均值，计数单位；均值非负。'],[raw`P_{pa},\quad I_a`,'反射 a 在像素 p 的轮廓份额（无量纲）及完整轮廓强度（计数）。'],[raw`Y_p,\ o_p,\ g_p`,'原始 ADU、读出偏置 ADU、增益 ADU/计数；oₚ 不同于占有率 oⱼ。'],[raw`\varepsilon_p,\ v_{r,p}`,'零均值读出噪声及其方差，分别为 ADU 和 ADU²。']])},
    {title:'推导：增益与背景决定像素方差',body:`${eq(raw`N_p\sim\operatorname{Poisson}(\ell_p),\quad \ell_p=b_p+\sum_aP_{pa}I_a`)}<p class="definition-line">ℓₚ 为总期望计数；Var 表示方差，假设计数与读出噪声独立。</p>${eq(raw`\mathbb E[Y_p]=o_p+g_p\ell_p,\qquad\operatorname{Var}(Y_p)=g_p^2\ell_p+v_{r,p}`)}${eq(raw`x_p=\frac{Y_p-o_p}{g_p}\Rightarrow\operatorname{Var}(x_p)=\ell_p+\frac{v_{r,p}}{g_p^2}`)}${points('两项分别来自计数涨落和读出噪声。','真实 DIALS 的 gain/pedestal 校正约定应按接口还原，不把单位转换等同于重新估计物理增益。')}`,cls:'derivation'},
  ]);
  replace('原始像素与自校准检测',`${symbols([[raw`H_p,\ m_p`,'检测核权重（无量纲）、有效像素掩膜（0 或 1）。'],[raw`x_p,\ b_p^{(x)}`,'用于检测的像素及匹配单位下的背景估计。'],[raw`\sigma_N^2,\quad Z`,'加权残差和的传播方差、无量纲标准化检测分数。']])}${eq(raw`N_{\mathrm{filt}}=\sum_pH_pm_p(x_p-b_p^{(x)}),\qquad Z=\frac{N_{\mathrm{filt}}}{\sigma_N}`)}${points('Nfilt 是过滤统计量，与光子计数 Nₚ 不同。','实际方差还校正像素相关、掩膜与背景估计误差。','检测临时还原 ADU 不回写 detector gain，不改变积分像素。')}`);
  after('原始像素与自校准检测',[
    {title:'推导：理想 matched filter 的权重',body:`${symbols([[raw`\mathbf r=\mathbf x-\mathbf b,\quad\Sigma`,'局部背景扣除向量及其固定噪声协方差，单位分别为像素值和其平方。'],[raw`\mathbf p,\ a_0,\ \mathbf w`,'已知形状向量、信号幅度、线性过滤权重；本页 p 是轮廓而非像素编号。']])}${eq(raw`\mathbb E[\mathbf r]=a_0\mathbf p,\qquad \mathrm{SNR}^2=a_0^2\frac{(\mathbf w^{\mathsf T}\mathbf p)^2}{\mathbf w^{\mathsf T}\Sigma\mathbf w}`)}${eq(raw`\mathbf w\propto\Sigma^{-1}\mathbf p,\qquad Z_*=\frac{\mathbf p^{\mathsf T}\Sigma^{-1}\mathbf r}{\sqrt{\mathbf p^{\mathsf T}\Sigma^{-1}\mathbf p}}`)}${points('对正定 Σ 白化后应用 Cauchy–Schwarz 不等式，等号给出最优权重。','Ariadne 的固定核与局部方差为工程近似，不是逐窗完整白化。')}`,cls:'derivation'},
  ]);
  before('重叠反射：可估计的强度方向',[
    {title:'联合积分：矩阵的行、列与背景',body:`${symbols([[raw`\mathbf y\in\mathbb R^m,\quad\mathbf I\in\mathbb R^K`,'m 个像素观测、K 个反射强度向量，须使用一致单位。'],[raw`P\in\mathbb R^{m\times K},\ P_a`,'轮廓设计矩阵；Pₐ 为第 a 列，即该反射的像素贡献形状。'],[raw`B_{\mathrm{bg}},\ \boldsymbol\beta`,'背景设计矩阵及背景系数，使乘积与 y 单位一致。'],[raw`\boldsymbol\varepsilon,\ \Sigma_y`,'零均值观测误差及其协方差；Σᵧ 的单位为观测单位平方。']])}${eq(raw`\mathbf y=P\mathbf I+B_{\mathrm{bg}}\boldsymbol\beta+\boldsymbol\varepsilon`)}<p class="footnote">Bbg 与晶胞矩阵 B 不同；动画中的 σ 表示合成 Gaussian 轮廓的宽度（像素），不是观测标准差 σₕ。</p>`},
  ]);
  after('重叠反射：可估计的强度方向',[
    {title:'推导：完全重叠时强度以总和进入预测',body:`${symbols([[raw`\mathbf v=(1,-1)^{\mathsf T}`,'两强度之间的差分方向。'],[raw`\delta`,'任意标量强度增量，只要变化后仍在允许参数域内。']])}${eq(raw`P=[P_1\ P_1]\quad\Rightarrow\quad P\mathbf v=0`)}${eq(raw`P\begin{pmatrix}I_1+\delta\\I_2-\delta\end{pmatrix}=P_1(I_1+I_2)`)}${points('沿差分方向改变参数不改变像素预测，所以差分不可辨识。','总和是否可估计还取决于轮廓与背景能否区分；非负约束不新增观测。','若引入结构先验来分离，两分量结果必须标为模型辅助。')}`,cls:'derivation'},
  ]);
  before('Gaussian 问题的 QR 压缩',[
    {title:'QR 压缩：先定义白化与增广系统',body:`${symbols([[raw`W=\Sigma_y^{-1}`,'固定正定误差协方差的逆，称权矩阵。'],[raw`D=W^{1/2}P,\quad\mathbf b=W^{1/2}\mathbf y`,'教学上假设背景已知并扣除；白化后的设计矩阵与观测。'],[raw`\mathbf z\in\mathbb R^K`,'本节待拟合的 K 个强度参数；不是前节密度或结构独立变量。'],[raw`Q,\ R`,'正交变换矩阵（QᵀQ=Iₘ）及增广矩阵的上梯形因子。']])}${eq(raw`[D\ \mathbf b]\in\mathbb R^{m\times(K+1)}`)}<p class="footnote">W¹ᐟ² 为满足 (W¹ᐟ²)ᵀW¹ᐟ²=W 的白化因子；本课用对称平方根表示。m 较小时不需压缩。</p>`},
  ]);
  after('Gaussian 问题的 QR 压缩',[
    {title:'推导：正交变换为何保留绝对残差',body:`${symbols([[raw`\mathbf v=\begin{bmatrix}\mathbf z\\-1\end{bmatrix}`,'长度 K+1 的增广参数向量；末项固定为 −1。']])}${eq(raw`D\mathbf z-\mathbf b=[D\ \mathbf b]\mathbf v=Q\begin{bmatrix}R\\0\end{bmatrix}\mathbf v`)}${eq(raw`\|D\mathbf z-\mathbf b\|^2=\mathbf v^{\mathsf T}\begin{bmatrix}R\\0\end{bmatrix}^{\mathsf T}Q^{\mathsf T}Q\begin{bmatrix}R\\0\end{bmatrix}\mathbf v=\|R\mathbf v\|^2`)}${points('保留增广最后一列，就保留无法被参数解释的常数残差。','同一 z 对应同一目标值，可行域 z≥0 也保持；未作秩截断。','仅在固定方差 Gaussian 平方残差目标下成立。')}`,cls:'derivation'},
  ]);
  before('结构反馈：模型信息与实验信息分开',[
    {title:'结构预测的不确定度：对角项与低秩项',body:`${symbols([[raw`\mathbf m,\ C_m`,'结构预测强度的均值和协方差，单位分别为强度与强度平方。'],[raw`U\in\mathbb R^{K\times r},\ \mathbf z\sim\mathcal N(\mathbf0,I_r)`,'低秩强度因子及 r 维无量纲隐变量；U 不是原子位移矩阵 Uⱼ。'],[raw`\boldsymbol\delta\sim\mathcal N(\mathbf0,D)`,'与 z 独立的逐反射误差；D 为强度方差对角矩阵，不是 QR 设计矩阵。']])}${eq(raw`\mathbf I=\mathbf m+U\mathbf z+\boldsymbol\delta`)}${eq(raw`\operatorname{Cov}(\mathbf I)=U I_rU^{\mathsf T}+D=UU^{\mathsf T}+D=C_m`)}`,cls:'derivation'},
    {title:'信息上限：矩阵不等式的方向含义',body:`${symbols([[raw`H_D`,'局部 Gaussian 数据目标的 Hessian / 信息矩阵，强度⁻²。'],[raw`\lambda_{\mathrm{eff}}\ge0,\quad\eta\ge0`,'先验有效权重和相对数据的信息预算，均无量纲。'],[raw`M\preceq N`,'半正定序：任意实向量 v 都满足 vᵀMv≤vᵀNv。']])}${eq(raw`\lambda_{\mathrm{eff}}C_m^{-1}\preceq\eta H_D`)}${eq(raw`\mathbf v^{\mathsf T}H_D\mathbf v=0\Rightarrow\mathbf v^{\mathsf T}\lambda_{\mathrm{eff}}C_m^{-1}\mathbf v=0`)}${points('若 Cₘ 正定且 Hᴅ 有非零零空间，严格全空间上限要求 λeff=0。','默认 η=0.25 是工程选择；可辨识方向、可行权重和实现边界须检查。')}`,cls:'derivation'},
  ]);
  before('已有数值证据：按实验口径解释',[
    {title:'检测指标：precision、recall 与 F1',body:`${symbols([[raw`\mathrm{TP},\ \mathrm{FP},\ \mathrm{FN}`,'按既定匹配规则计数：真阳性、假阳性、假阴性。'],[raw`\mathrm{Precision},\ \mathrm{Recall}`,'预测中的真阳性比例、参考真信号中被找回的比例，均无量纲。']])}${eq(raw`\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\quad\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`)}${eq(raw`F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}`)}${points('分母为零时须采用明确评价约定；不能默认为完美检测。','不完整真实参考不能枚举全部真信号，支持率不自动等于 precision。')}`},
  ]);
  // 用内容角色而非页码控制排版，新增教学页不会把样式应用到错误章节。
  lookup('七晶系：晶胞形状的并列比较').cls='systems-overview';
  lookup('Ariadne：从原始帧到结构证据').cls='pipeline-overview';
  lookup('Bragg 定律与实验分辨率').cls='resolution-example';
  slides.forEach(s=>{s.body=s.body.replace(/32 页课件 PDF/g,'课件 PDF');});
})();
