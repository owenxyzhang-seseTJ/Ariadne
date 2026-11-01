'use strict';
window.LectureFigures = (() => {
  const label=(x,y,t,small=false)=>`<text x="${x}" y="${y}" class="${small?'fig-small':'fig-text'}">${t}</text>`;
  const line=(x,y,X,Y,cls='')=>`<path d="M${x},${y}L${X},${Y}" class="${cls}"/>`;
  const dot=(x,y,r=3,open=false)=>`<circle cx="${x}" cy="${y}" r="${r}" class="${open?'open':'filled'}"/>`;
  const svg=(body,title,view='0 0 560 350')=>`<svg viewBox="${view}" role="img" aria-label="${title}" class="textbook"><title>${title}</title><defs><marker id="arrow-${svg.counter++}"/></defs>${body}</svg>`;
  svg.counter=0;
  const arrow=(x,y,X,Y)=>{
    const a=Math.atan2(Y-y,X-x),s=7;
    return line(x,y,X,Y)+`<path d="M${X-s*Math.cos(a-.4)},${Y-s*Math.sin(a-.4)}L${X},${Y}L${X-s*Math.cos(a+.4)},${Y-s*Math.sin(a+.4)}"/>`;
  };
  function motif(){
    let b='';
    for(let y=0;y<3;y++)for(let x=0;x<4;x++){
      const X=25+x*55+y*10,Y=75+y*82;
      b+=dot(X,Y,3);
      const xx=X+300;
      b+=line(xx,Y,xx+19,Y-12)+line(xx+19,Y-12,xx+32,Y+8)+dot(xx,Y,5)+dot(xx+19,Y-12,5,true)+dot(xx+32,Y+8,3);
    }
    b+=arrow(228,170,286,170)+label(35,327,'点阵：等价环境的位置')+label(325,327,'点阵 + 同一结构基元');
    b+=`<path d="M25 75h55l10 82H35Z" class="heavy"/>`;
    return svg(b,'二维点阵与三原子结构基元的周期重复');
  }
  const cellPoints=(x,y,w,h,skew,depth)=>[[x,y],[x+w,y],[x+w+skew,y-depth],[x+skew,y-depth],[x,y-h],[x+w,y-h],[x+w+skew,y-h-depth],[x+skew,y-h-depth]];
  function cellLines(p){return [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]].map(([a,b])=>line(...p[a],...p[b],[[2,3],[3,0],[3,7]].some(e=>e[0]===a&&e[1]===b)?'hidden-edge':'')).join('');}
  function cell(){
    const p=cellPoints(100,275,235,175,100,65);
    return svg(cellLines(p)+arrow(100,275,360,275)+arrow(100,275,218,198)+arrow(100,275,100,63)+label(365,286,'a')+label(222,195,'b')+label(83,57,'c')+label(212,306,'a')+label(392,255,'b')+label(446,139,'c')+`<path d="M141 275A41 41 0 0 0 134 253M100 241A34 34 0 0 1 129 256M100 228A47 47 0 0 1 141 249"/>`+label(148,265,'γ')+label(110,231,'α')+label(76,235,'β')+dot(100,275,4)+label(67,299,'O'),'一般平行六面体晶胞，直接基矢与晶胞参数');
  }
  function primitive(){
    let b='';for(let x=0;x<5;x++)for(let y=0;y<4;y++){const X=70+x*87,Y=45+y*80;b+=dot(X,Y)+dot(X+43.5,Y+40,3,true);}
    b+=`<path d="M157 125H244V205H157Z" class="heavy"/><path d="M244 125l43.5 40 43.5-40-43.5-40Z" class="heavy"/>`;
    b+=label(130,321,'矩形惯用胞：2 个点阵点')+label(320,321,'菱形原胞：1 个点阵点');
    return svg(b,'二维中心矩形点阵中的惯用胞与原胞');
  }
  function families(){
    const shapes=[['三斜',70,90,61,70,24,22],['单斜',235,90,64,80,25,20],['正交',410,90,70,85,27,23],['四方',70,250,70,95,33,18],['六方',235,250,70,100,-28,23],['立方',410,250,75,75,34,25]];
    return svg(shapes.map(([n,x,y,w,h,s,d])=>cellLines(cellPoints(x,y,w,h,s,d))+label(x,y+30,n)).join(''),'六晶族的晶胞示意；图形仅示意常见惯用设定','0 0 560 320');
  }
  function systems(){
    const types=[['三斜',1,1.2,1.35,74,81,68],['单斜',1,1.2,1.4,90,108,90],['正交',1,1.2,1.5,90,90,90],['四方',1,1,1.5,90,90,90],['三方 · 菱方轴',1.15,1.15,1.15,76,76,76],['六方 · 六方轴',1,1,1.5,90,90,120],['立方',1.15,1.15,1.15,90,90,90]];
    let out='';
    types.forEach(([name,a,b,c,al,be,ga],i)=>{
      const rad=Math.PI/180,co=t=>Math.cos(t*rad),si=t=>Math.sin(t*rad);
      const cy=c*(co(al)-co(be)*co(ga))/si(ga),cz=Math.sqrt(c*c-(c*co(be))**2-cy*cy);
      const v=[[a,0,0],[b*co(ga),b*si(ga),0],[c*co(be),cy,cz]],col=i<4?i:i-4,row=i<4?0:1;
      const X=col*165+45+(row?80:0),Y=row*185+150;
      const project=(u,w,z)=>[X+52*(u+.8*w),Y-52*(z+.5*w)];
      const pts=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]].map(s=>project(...[0,1,2].map(k=>s.reduce((n,t,j)=>n+t*v[j][k],0))));
      out+=cellLines(pts)+label(X-14,Y+29,name);
    });
    return svg(out,'七晶系的晶胞投影，以相同斜投影显示三斜、单斜、正交、四方、三方、六方与立方','0 0 700 390');
  }
  function centering(){
    const offsets={P:[],I:[[.5,.5,.5]],F:[[.5,.5,0],[.5,.5,1],[.5,0,.5],[.5,1,.5],[0,.5,.5],[1,.5,.5]],C:[[.5,.5,0],[.5,.5,1]]};
    let b='';['P','I','F','C'].forEach((type,i)=>{
      const x=30+i*137,y=172,w=72,h=94,s=35,d=27,p=cellPoints(x,y,w,h,s,d);
      b+=cellLines(p);p.forEach(q=>{b+=dot(...q,2.5);});
      offsets[type].forEach(([a,c,z])=>{b+=dot(x+a*w+c*s,y-z*h-c*d,5,true);});
      b+=label(x+35,235,type)+label(x+5,264,({P:'1 点 / 胞',I:'2 点 / 胞',F:'4 点 / 胞',C:'2 点 / 胞'})[type],true);
    });return svg(b,'P、I、F、C 点阵的惯用胞，空心大圆表示附加点阵点','0 0 580 290');
  }
  function miller(){
    const x=75,y=273,a=[305,0],b=[105,-95],c=[0,-190];
    const p=(u,v,w)=>[x+u*a[0]+v*b[0]+w*c[0],y+u*a[1]+v*b[1]+w*c[1]];
    let s=arrow(...p(0,0,0),...p(1.15,0,0))+arrow(...p(0,0,0),...p(0,1.3,0))+arrow(...p(0,0,0),...p(0,0,1.17));
    const A=p(1,0,0),B=p(0,1,0),C=p(0,0,.5);
    s+=`<path d="M${A}L${B}L${C}Z" class="plane"/>`+label(...p(1.2,0,0),'a')+label(...p(0,1.4,0),'b')+label(...p(0,0,1.22),'c')+label(A[0]+3,A[1]+23,'a')+label(B[0]+12,B[1],'b')+label(C[0]-43,C[1],'c/2')+label(245,174,'(112)')+label(50,302,'O');
    return svg(s,'Miller 指数 (112)：截距 a、b、c/2，取倒数得 1、1、2');
  }
  function reciprocal(){
    let b='';for(let i=0;i<5;i++)for(let j=0;j<4;j++)b+=dot(35+i*42+j*12,250-j*55);
    b+=arrow(35,250,119,250)+arrow(35,250,59,140)+label(119,273,'a')+label(32,134,'b')+label(43,326,'直接点阵')+arrow(245,175,305,175);
    for(let i=0;i<5;i++)for(let j=0;j<4;j++)b+=dot(340+i*39,250-j*52+i*39*12/55);
    b+=arrow(340,250,418,250+78*12/55)+arrow(340,250,340,146)+label(424,285,'a*')+label(307,140,'b*')+label(370,326,'倒易点阵');
    return svg(b,'直接与倒易基矢互为对偶，倒易基矢垂直相应直接晶面');
  }
  function structureFactor(){
    let b='';for(let i=0;i<5;i++){const x=50+i*100;b+=line(x,180,x+26,144)+dot(x,180,7)+dot(x+26,144,5,true);}
    b+=label(33,230,'周期平移决定反射位置')+label(298,230,'胞内基元决定反射强度')+arrow(50,180,150,180)+label(88,205,'a');
    return svg(b,'含两个不等价散射中心的周期结构，点阵与基元分别影响衍射位置和强度');
  }
  function symmetry(){
    let b='';for(let i=0;i<4;i++){const a=i*Math.PI/2,X=150+65*Math.cos(a),Y=160+65*Math.sin(a);b+=dot(X,Y,7)+line(150,160,X,Y);}b+=dot(150,160,3,true)+label(122,265,'四重旋转');
    b+=line(380,50,380,250,'hidden-edge')+`<path d="M325 100l-40 90h65Z M435 100l40 90h-65Z"/>`+label(337,265,'镜面反射');
    return svg(b,'点对称操作：四重旋转和镜面反射');
  }
  function phase(){
    return svg(`${arrow(70,175,70,45)}${arrow(70,175,255,175)}<circle cx="70" cy="175" r="105" class="hidden-edge"/><path d="M70 175L157 116" class="heavy"/><path d="M111 175A41 41 0 0 0 104 152"/>${label(121,163,'φₕ')}${label(128,112,'|Fₕ|')}${label(233,200,'Re')}${label(40,35,'Im')}${label(325,122,'实验强度 → 模长')}${label(325,172,'缺失相位 → 逆问题')}${label(325,222,'原子性、组成、对称性',true)}`, '复平面中结构因子模长与相位的分解');
  }
  function pipeline(){
    const names=['像素','检测','索引','积分','求解','精修'];let b='';names.forEach((n,i)=>{const x=20+i*90;b+=`<rect x="${x}" y="100" width="72" height="50"/>`+label(x+16,132,n);if(i<5)b+=arrow(x+72,125,x+88,125);});b+=label(40,214,'仪器与噪声')+label(217,214,'几何与统计')+label(411,214,'结构与化学');return svg(b,'原始像素至结构精修的有序计算链','0 0 570 270');
  }
  function qr(){
    let b='';for(let r=0;r<17;r++)for(let c=0;c<5;c++)b+=`<rect x="${55+c*18}" y="${20+r*16}" width="13" height="10" fill="${c===4?'#888':'#ddd'}" stroke="none"/>`;
    b+=arrow(190,150,310,150)+label(234,128,'QR');for(let r=0;r<5;r++)for(let c=r;c<5;c++)b+=`<rect x="${350+c*23}" y="${92+r*23}" width="18" height="18" fill="${c===4?'#888':'#ddd'}" stroke="none"/>`;
    return svg(b+label(44,331,'m 个像素方程')+label(345,268,'至多 K+1 行'),'增广矩阵的 QR 压缩，末列观测保留常数残差');
  }
  function agent(){
    return svg(`<rect x="30" y="20" width="500" height="55"/>${label(156,54,'目标、输入、预算与接受条件')}${arrow(180,75,180,105)}${arrow(400,75,400,105)}<rect x="30" y="105" width="275" height="60"/><rect x="335" y="105" width="195" height="60"/>${label(65,142,'确定性计划器 / 任务 DAG')}${label(362,142,'可选语言模型')}${arrow(335,135,305,135)}${arrow(180,165,180,203)}<rect x="30" y="203" width="500" height="55"/>${label(100,237,'有类型的工具：计算、产物、诊断')}${arrow(280,258,280,288)}${label(100,319,'来源记录 → 独立检查 → 接受或分支')}`, '计划层、可选语言模型、数值工具与证据检查的分层架构');
  }
  function feedback(){
    return svg(`<rect x="25" y="45" width="205" height="70"/><rect x="330" y="45" width="205" height="70"/>${label(75,87,'像素似然')}${label(380,87,'结构预测')}${arrow(230,65,330,65)}${arrow(330,100,230,100)}${arrow(125,115,125,205)}${arrow(430,115,430,205)}<rect x="25" y="205" width="205" height="70"/><rect x="330" y="205" width="205" height="70"/>${label(75,246,'信息上限')}${label(366,246,'分组留出检验')}`, '结构反馈由数据方向的信息上限和独立分组留出共同约束');
  }
  function drawBragg(canvas,thetaDeg=30,t=0){
    const a=prepare(canvas);if(!a)return;const {ctx,w,h}=a;ctx.save();ctx.scale(w/560,h/350);
    const A=[275,130],B=[275,230],d=100,th=thetaDeg*Math.PI/180,c=Math.cos(th),s=Math.sin(th),u=[c,s],v=[c,-s];
    const incoming=[B[0]-d*s*c,B[1]-d*s*s],outgoing=[B[0]+d*s*c,B[1]-d*s*s];
    const seg=(p,q,width=1,dash=[])=>{ctx.beginPath();ctx.lineWidth=width;ctx.setLineDash(dash);ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();ctx.setLineDash([]);};
    const txt=(p,v,size=17)=>{ctx.font=`${size}px 'Times New Roman',serif`;ctx.fillText(v,...p);};
    ctx.strokeStyle='#222';ctx.fillStyle='#222';[130,230,310].forEach(y=>seg([35,y],[525,y]));
    for(const p of [A,B]){seg([p[0]-170*c,p[1]-170*s],p,1.4);seg(p,[p[0]+190*c,p[1]-190*s],1.4);}
    seg(incoming,B,4);seg(B,outgoing,4);seg(A,incoming,1,[4,4]);seg(A,outgoing,1,[4,4]);
    seg([50,130],[50,230]);seg([44,130],[56,130]);seg([44,230],[56,230]);txt([28,185],'d');
    for(let x=80;x<=500;x+=50){ctx.beginPath();ctx.arc(x,130,2,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(x,230,2,0,Math.PI*2);ctx.fill();}
    txt([A[0]-8,A[1]-13],'A');txt([B[0]-8,B[1]+24],'B');txt([150,207],'d sin θ',14);txt([337,207],'d sin θ',14);txt([450,118],'晶面',14);
    ctx.beginPath();ctx.arc(...A,38,Math.PI,Math.PI+th);ctx.stroke();txt([218,118],'θ');
    ctx.beginPath();ctx.arc(...A,64,-th,th);ctx.stroke();txt([A[0]+71,A[1]+5],'2θ');
    // 同相波包分别沿两条反射路径传播，粗线段给出附加光程。
    for(const p of [A,B])for(let i=0;i<5;i++){
      const L=((t*.065+i*65)%340)-170;const q=L<0?[p[0]+L*c,p[1]+L*s]:[p[0]+L*c,p[1]-L*s];
      seg([q[0]-4*s,q[1]+4*c],[q[0]+4*s,q[1]-4*c],1.5);
    }
    txt([40,340],`θ = ${thetaDeg.toFixed(1)}°；Δ / d = ${(2*s).toFixed(3)}`,15);ctx.restore();
  }
  function prepare(canvas){const r=canvas.getBoundingClientRect();if(!r.width||!r.height)return null;const d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);const ctx=canvas.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,r.width,r.height);return{ctx,w:r.width,h:r.height};}
  function drawEwald(canvas,phiDeg=0){
    const a=prepare(canvas);if(!a)return;const {ctx,w,h}=a;ctx.save();ctx.scale(w/560,h/380);
    const O=[125,215],C=[275,215],R=150,phi=phiDeg*Math.PI/180;
    const p=(i,j)=>[O[0]+60*(i*Math.cos(phi)-j*Math.sin(phi)),O[1]+60*(i*Math.sin(phi)+j*Math.cos(phi))];
    const seg=(x,y,dashed=false)=>{ctx.beginPath();ctx.setLineDash(dashed?[5,4]:[]);ctx.moveTo(...x);ctx.lineTo(...y);ctx.stroke();ctx.setLineDash([]);};
    const txt=(x,y,v,size=16)=>{ctx.font=`${size}px 'Times New Roman',serif`;ctx.fillText(v,x,y);};
    ctx.strokeStyle='#222';ctx.fillStyle='#222';ctx.lineWidth=1.2;ctx.beginPath();ctx.arc(...C,R,0,Math.PI*2);ctx.stroke();
    for(let i=-2;i<=6;i++)for(let j=-4;j<=3;j++){
      const P=p(i,j);if(P[0]<20||P[0]>540||P[1]<15||P[1]>365)continue;
      const on=Math.abs(Math.hypot(P[0]-C[0],P[1]-C[1])-R)<2;
      ctx.beginPath();ctx.arc(...P,on?4:2,0,Math.PI*2);ctx.fillStyle=on?'#111':'#aaa';ctx.fill();
    }
    const P=p(1,-2),err=(Math.hypot(P[0]-C[0],P[1]-C[1])-R)/R;
    ctx.fillStyle='#111';ctx.strokeStyle='#111';ctx.lineWidth=1.7;
    const arr=(a,b,dash=false)=>{seg(a,b,dash);const t=Math.atan2(b[1]-a[1],b[0]-a[0]);seg([b[0]-8*Math.cos(t-.35),b[1]-8*Math.sin(t-.35)],b);seg(b,[b[0]-8*Math.cos(t+.35),b[1]-8*Math.sin(t+.35)]);};
    arr(C,O);arr(O,P);arr(C,P,Math.abs(err)>1e-8);
    ctx.beginPath();ctx.arc(...O,5,0,Math.PI*2);ctx.fillStyle='white';ctx.fill();ctx.stroke();ctx.fillStyle='#111';
    ctx.beginPath();ctx.arc(...P,5,0,Math.PI*2);Math.abs(err)<1e-8?ctx.fill():ctx.stroke();
    txt(103,240,'O');txt(281,237,'C');txt(179,238,'s₀');txt(P[0]-24,P[1]-13,'H');txt((O[0]+P[0])/2-24,(O[1]+P[1])/2,'q');txt((C[0]+P[0])/2+10,(C[1]+P[1])/2,Math.abs(err)<1e-8?'s₁':'s₀ + q');txt(394,110,'1/λ');
    txt(28,374,`φ = ${phiDeg.toFixed(1)}°；径向失配 / 球半径 = ${err.toFixed(4)}`,14);ctx.restore();return err;
  }
  function drawOverlap(canvas,separation=2.6){
    const a=prepare(canvas);if(!a)return;const {ctx}=a;ctx.save();ctx.scale(a.w/560,a.h/350);const w=560,h=350;ctx.strokeStyle='#aaa';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(35,18);ctx.lineTo(35,h-35);ctx.lineTo(w-12,h-35);ctx.stroke();
    const curves=[[],[],[]];for(let i=0;i<=240;i++){const x=35+(w-50)*i/240,t=(i/240-.5)*12,p=Math.exp(-.5*(t+separation/2)**2),q=.72*Math.exp(-.5*(t-separation/2)**2);[p,q,p+q].forEach((v,j)=>curves[j].push([x,h-35-v*(h-65)/1.75]));}
    curves.forEach((pts,j)=>{ctx.beginPath();ctx.strokeStyle='#111';ctx.lineWidth=j===2?2:1.3;ctx.setLineDash(j===0?[7,4]:j===1?[2,4]:[]);pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();});ctx.setLineDash([]);ctx.font='12px serif';ctx.fillStyle='#333';ctx.fillText('相对幅度',6,12);ctx.fillText('像素位置',w-70,h-10);ctx.restore();
  }
  return {svg,label,line,dot,arrow,motif,cell,primitive,families,systems,centering,miller,reciprocal,structureFactor,symmetry,phase,pipeline,qr,agent,feedback,drawBragg,drawEwald,drawOverlap};
})();
