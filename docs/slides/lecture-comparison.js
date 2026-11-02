'use strict';
// 按照工作区对比文档补充检测方法、数据口径和限制。
(() => {
  const slides=window.LECTURE;
  const points=(...items)=>`<ul class="points">${items.map(s=>`<li>${s}</li>`).join('')}</ul>`;
  const table=(heads,rows)=>`<div class="table-wrap"><table><thead><tr>${heads.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const source='<p class="footnote">资料来源：ARIADNE_VS_DIALS.zh-CN.md，2026年11月已有记录；本次制作课件未重跑实验。</p>';
  const insert=(title,items)=>{const i=slides.findIndex(s=>s.title===title);if(i<0)throw Error(`缺少课件页：${title}`);slides.splice(i,0,...items.map(s=>({...s,part:'Ⅴ · Ariadne 与 DIALS 对照',notes:s.notes||'本页比较的具体实验条件和限制见正文；既有实验并非本次重跑。'})));};
  insert('已有数值证据：按实验口径解释',[
    {title:'Ariadne 与 DIALS：处理路线',body:`${table(['环节','Ariadne 方案','本次 DIALS 对照'],[['斑点检测','匹配滤波聚合邻域信号，自校准背景与有效噪声','Extended：局部离散度及强度阈值'],['数据约化','检测输出接入 DIALS 的索引、几何精修和积分','DIALS 负责数据约化'],['主要取舍','所测弱峰与部分近邻峰场景找回更多候选','所测检测配置耗时较短']])}${points('比较对象是特定配置的 DIALS Extended，并非 DIALS 全部算法。','检测质量和速度的比较不能直接替代最终结构准确率比较。')}${source}`},
    {title:'合成留出集：检测召回与精确率',body:`${table(['场景 / 方法','Precision','Recall','F1'],[['规则 / Ariadne','99.85%','69.29%','0.8181'],['规则 / DIALS Extended','100.00%','36.33%','0.5329'],['复杂 / Ariadne','98.93%','63.88%','0.7763'],['复杂 / DIALS Extended','99.69%','44.01%','0.6106']])}<p class="definition-line">Precision：预测斑点中真阳性的比例；Recall：参考真斑点被找回的比例；F1：二者的调和平均。</p>${points('每场景 20 张 512×512 测试图；双方均去除小于 6 像素的组件并按位置一对一匹配。','DIALS 对照使用真实背景和读出方差构造的噪声方差图；测试集未用于局部参数选择。')}${source}`},
    {title:'真实帧：已有反射的参考召回',body:`${table(['检测方法','I/σ ≥ 3 的参考召回','参考支持比例'],[['Ariadne matched','96.66%','83.45%'],['DIALS Extended：原 panel gain','65.20%','90.79%'],['DIALS Extended：有效增益校准','80.46%','89.94%'],['DIALS Extended：增益与基线校准','84.75%','88.88%']])}${points('5 个扫描各取 8 帧；参考来自既有双域积分反射，采用相同 underflow 解释和组件下限。','参考表不是完整真值，参考支持比例不等于实际 precision；重叠也可能影响匹配。')}${source}`},
    {title:'近邻双峰：峰间距决定分离难度',body:`${table(['峰间距','Ariadne 分开峰对','DIALS Extended 分开峰对'],[['2 像素','0 / 150','0 / 150'],['4 像素','25 / 150','6 / 150'],['6 像素','117 / 150','72 / 150'],['8 像素','124 / 150','87 / 150']])}${points('合成 Gaussian 峰宽 σ = 1.1 像素，强度比为 1:1 或 1:4；含背景梯度与计数、读出噪声。','一对一匹配避免把一个合并组件算作两个峰；2 像素时两种方法均未分开。','检测出两个候选不等于已把孪晶重叠反射的强度拆分。')}${source}`},
    {title:'同轮检测计时：质量与速度的取舍',body:`${table(['方法','中位耗时 / 帧'],[['Ariadne 优化前，四工作单元','36.11 ms'],['Ariadne 优化后，四工作单元','24.96 ms'],['DIALS Extended：复用 C++ / flex 输入','11.64 ms'],['DIALS Extended：含 NumPy / flex 转换','13.88 ms']])}${points('4 张真实帧，同进程交替运行 26 轮，剔除各方法最初 2 次样本。','优化后相对同轮旧内核延迟降低 30.9%；相对 DIALS C++ 输入耗时约 2.14 倍。','计时排除磁盘读取与径向 bins 生成；不是整链吞吐量，也不是等召回率下的最优速度比较。')}<p class="footnote">数据来自未重新打包 wheel 的工作区内核；此结果不可直接归属于已发布 wheel。</p>${source}`},
    {title:'如何选择与解读检测结果',body:`${points('弱衍射、结构化背景和实际噪声与标称参数不一致时，可评估自校准与匹配滤波的收益。','清晰、分离良好的强峰及低延迟需求下，本次 DIALS Extended 对照更快。','32 帧真实路线执行至积分并得到 1,048 行输出；这验证接口衔接，不证明积分或最终结构更准确。','完全重叠可能只支持强度组合；真实孪晶端到端结构解析仍需独立验证。')}${source}`},
  ]);
  const old=slides.find(s=>s.title==='已有数值证据：按实验口径解释');
  old.body=old.body.replace('87.195 ms','87.195 ms（旧轮次）').replace('27.950 ms','27.950 ms（旧轮次）').replace('检测耗时（四工作单元）','历史检测耗时（四工作单元）').replace('速度仍慢约 3.12 倍','旧轮次速度慢约 3.12 倍，不与新轮次直接计算加速比');
})();
