'use strict';
// 核心算法专题插入现有基础课之后，保持图文与符号约定一致。
(() => {
  const slides=window.LECTURE, F=window.LectureFigures, raw=String.raw;
  const math=(tex,inline=false)=>`<${inline?'span':'div'} class="${inline?'math-inline':'math'}" data-tex="${tex.replaceAll('&','&amp;').replaceAll('"','&quot;')}"></${inline?'span':'div'}>`;
  const eq=t=>math(t), mi=t=>math(t,true);
  const points=(...items)=>`<ul class="points">${items.map(t=>`<li>${t}</li>`).join('')}</ul>`;
  const defs=rows=>`<dl class="symbols">${rows.map(([s,d])=>`<div><dt>${mi(s)}</dt><dd>${d}</dd></div>`).join('')}</dl>`;
  const split=(a,b)=>`<div class="lecture-split"><div>${a}</div><div>${b}</div></div>`;
  const fig=(x,c)=>`<figure>${x}<figcaption>${c}</figcaption></figure>`;
  const insert=(title,items)=>{const i=slides.findIndex(s=>s.title===title);if(i<0)throw Error(`缺少课件页：${title}`);slides.splice(i,0,...items.map(s=>({...s,part:'Ⅳ · 核心算法与反馈',notes:s.notes||'本页说明算法思想；实际适用范围和证据以课件中的验证说明为准。'})));};
  const loop=F.svg(`<rect x="18" y="34" width="140" height="54"/><rect x="210" y="34" width="140" height="54"/><rect x="402" y="34" width="140" height="54"/><rect x="402" y="225" width="140" height="54"/><rect x="210" y="225" width="140" height="54"/><rect x="18" y="225" width="140" height="54"/>${F.label(48,67,'原始像素')}${F.label(238,67,'检测 / 积分')}${F.label(431,67,'结构候选')}${F.label(431,258,'精修 / 检查')}${F.label(238,258,'结构预测')}${F.label(46,258,'反馈 / 再判断')}${F.arrow(158,61,207,61)}${F.arrow(350,61,399,61)}${F.arrow(472,88,472,222)}${F.arrow(402,252,353,252)}${F.arrow(210,252,161,252)}${F.arrow(88,225,88,91)}`,'数据约化、结构候选、精修、结构预测和反馈构成可审计闭环','0 0 560 320');
  insert('Ariadne：从原始帧到结构证据',[
    {title:'Ariadne：自反馈 agent 结构解析框架',body:split(`${points('以任务图连接数据约化、结构求解、精修和结构反馈。','候选结构预测反射强度及不确定度，反馈到重叠积分或后验推断。','agent 保存依赖、产物与诊断，根据检查结果选择继续、分支或暂停。','闭环由数值工具执行；语言模型是可选的计划与解释组件。')}<p class="footnote">“自反馈”描述结构模型与数据处理之间的信息回路，不意味着无需人工确认或任何晶体都能自动求解。</p>`,fig(loop,'图 1　可审计的结构反馈闭环；教学示意。'))},
    {title:'算法组合：每一步解决什么问题',body:`<div class="table-wrap"><table><thead><tr><th>挑战</th><th>核心方法</th><th>输出与判据</th></tr></thead><tbody><tr><td>弱衍射、非均匀背景</td><td>自校准背景与噪声；匹配滤波</td><td>候选斑点、校准分数</td></tr><tr><td>多晶畴反射重叠</td><td>几何预测、像素级联合轮廓拟合</td><td>强度组合、秩与不确定度</td></tr><tr><td>弱约束与模型依赖</td><td>结构预测的相关先验与信息上限</td><td>实验估计与模型辅助后验分开</td></tr><tr><td>多步骤计算</td><td>有类型工具、持久化任务图与分支</td><td>产物、来源、恢复与检查记录</td></tr></tbody></table></div><p class="footnote">上述方法是系统组合和实现设计；孪晶整链效果必须用具体数据单独评估。</p>`},
  ]);
  insert('原始像素与自校准检测',[
    {title:'弱衍射：为何先估计背景与噪声',body:`${defs([[raw`b_p=P_{r(p)}S_p`,'像素 p 的背景均值，分解为径向剖面 P 与空间调制 S；单位与 xₚ 一致。'],[raw`r(p),\ \alpha,\ \beta`,'r(p) 为径向分箱；α、β 描述背景残差方差随背景强度的变化。']])}${eq(raw`\operatorname{Var}(x_p-b_p)\approx\alpha b_p+\beta`)}${points('有效像素块上交替估计径向和空间背景，保留原始观测。','按背景强度分箱估计方差，校准有效增益、基线和读出噪声。','弱斑点可低于局部背景起伏；先校准再按轮廓累积证据。')}<p class="footnote">变量 P 在此是径向背景剖面，不是后续积分设计矩阵。</p>`},
    {title:'弱衍射：相关噪声与多尺度候选',body:`${defs([[raw`\rho_d,\ \kappa_H`,'背景像素间位移 d 的相关系数；κH 为检测核的相关噪声修正，均无量纲。'],[raw`H_p,\ m_p,\ Z,\ V`,'检测核权重、有效像素掩膜、标准化分数和残差传播方差；V 不是晶胞体积。']])}${eq(raw`Z=\frac{\sum_p H_pm_p(x_p-b_p)}{\sqrt{V}},\qquad V\ \text{包含}\ \kappa_H\ \text{与局部背景估计误差}`)}${points('背景像素估计多个位移的相关系数；相关性会改变加权和的方差。','细、粗两个尺度结合局部极大、watershed 和尖峰检查形成候选。','阈值依赖近似噪声模型；设定的 false_alarms 不等于已验证的实际误检率。')}`},
  ]);
  const twin=F.svg(`<circle cx="280" cy="175" r="7" class="filled"/>${[-2,-1,0,1,2].map(i=>[-2,-1,0,1,2].map(j=>F.dot(280+i*47,175+j*47,3)).join('')).join('')}${[-2,-1,0,1,2].map(i=>[-2,-1,0,1,2].map(j=>F.dot(280+i*41-j*23,175+i*23+j*41,4,true)).join('')).join('')}${F.label(38,42,'实心：晶畴 1')}${F.label(345,42,'空心：晶畴 2')}${F.label(175,330,'二维倒易点阵示意')}`, '两个取向不同的二维倒易点阵，部分节点发生重叠');
  insert('重叠反射：可估计的强度方向',[
    {title:'孪晶与多晶格：从几何候选到联合像素',body:split(`${points('多个晶畴各自产生倒易点阵；不同来源的反射可能落在同一探测器区域。','保留同一斑点的多个几何候选，再按共同扫描、帧和探测器区域构造重叠组。','对同一区域的所有贡献与背景联合拟合，而不是先把每个像素硬分配给某一晶畴。','几何相容度用于比较候选，不自动等于经过校准的来源概率或晶畴比例。')}<p class="footnote">默认原始帧 reduce 路线为单晶格；复杂孪晶需显式构建 custom 联合处理图。</p>`,fig(twin,'图 1　不同晶畴的倒易节点可在探测器上重叠；图为二维示意。'))},
    {title:'联合积分：保留弱反射与可辨识性',body:`${defs([[raw`\mathbf y,\ P,\ \mathbf I`,'共享区域内的像素、各反射轮廓矩阵及对应强度向量。'],[raw`B_{\mathrm{bg}}\boldsymbol\beta,\ W`,'局部背景模型与像素权矩阵；权重采用给定误差模型。']])}${eq(raw`\widehat{\mathbf I},\widehat{\boldsymbol\beta}=\arg\min_{\mathbf I,\boldsymbol\beta}\|W^{1/2}(\mathbf y-P\mathbf I-B_{\mathrm{bg}}\boldsymbol\beta)\|^2`)}${points('有符号加权最小二乘保留弱测量的负值，避免逐项截零造成偏差。','利用 QR 消去背景，SVD 分析秩与参数相关性。','输出 joint_components、total_only 或 unresolved，而非强行拆分不可辨识分量。')}<p class="footnote">本页是有符号 Gaussian 路径；非负 NNLS 与计数似然使用各自的约束和假设。</p>`},
    {title:'不同重叠几何：累积可观测信息',body:`${defs([[raw`\mathbf c_g,\ M_g,\ \boldsymbol\xi_g`,'第 g 个重叠组的可观测强度组合、映射矩阵和观测噪声。'],[raw`\mathbf I,\ M_{\mathrm{all}}`,'跨组共享的潜在反射强度，以及各组映射纵向拼接的矩阵。']])}${eq(raw`\mathbf c_g=M_g\mathbf I+\boldsymbol\xi_g,\qquad M_{\mathrm{all}}=\begin{bmatrix}M_1\\M_2\\\vdots\end{bmatrix}`)}${points('一个重叠组可能只能测量强度的某些线性组合；另一个几何可提供不同组合。','联合秩提高时，可恢复的强度方向也可能增加。','需正确关联晶畴、Miller 指标和共同像素来源；不能重复计入同一观测。')}<p class="footnote">固定轮廓、几何和权重的 Gaussian 模型下，信息模式可保留相应线性信息。</p>`},
    {title:'孪晶算法：现有证据与边界',body:`${points('已知轮廓、几何和噪声的三晶畴合成实验：两个局部秩 2 的几何联合达到秩 3。','同一合成设定中，压缩信息模式与原像素联合解的最大强度差为 2.27×10⁻¹³。','500 次模拟中，最弱幅度真值 0.8，估计均值 0.79295；弱测量出现负值并不自动代表错误。','上述结果是特定模拟与固定模型的证据，不证明未知孪晶的自动检出或真实结构精度。')}<p class="footnote">来源：仓库 FIRST_PRINCIPLES_UPDATE.md、reports/twin_information_experiment.json；本次未重跑实验。</p>`},
  ]);
  insert('agent 架构：计划、工具与证据分层',[
    {title:'自反馈：从结构候选到可检验的后验',body:split(`${points('对候选结构计算反射预测及其不确定度，并与积分所得的 Gaussian 信息组合。','按共享强度和等价关系将数据分组：训练校准、验证选候选、冻结后测试。','区分数据支持的方向和先验提供的信息；报告模型辅助后验与原始观测。','记录候选、参数、数据来源及停止或分支原因，允许任务暂停后恢复。')}<p class="footnote">候选结构预测目前基于实数 IT92 X 射线散射；固定 Gaussian 反馈不是统一的像素—原子坐标联合精修。</p>`,fig(F.feedback(),'图 1　结构预测、数据约束和分组检查。'))},
    {title:'自反馈与孪晶：接口和自动化边界',body:`${points('有符号 Gaussian 联合积分可通过显式桥接进入反馈；需要声明域、指标、强度身份、校正和原始像素来源。','桥接不会把 NNLS 或 Poisson 结果冒充无约束 Gaussian 观测；重复来源应在分组中闭合。','当前任务图可调用联合积分等工具，但默认单晶 reduce 不会自动完成真实孪晶全流程与反馈桥接。','已有候选结构和反馈的演示使用合成观测；真实孪晶端到端准确性仍待验证。')}<p class="footnote">“自反馈 agent”强调数据与结构模型的受控信息回路，而非宣称已实现无人监督的完整孪晶解析。</p>`},
  ]);
})();
