'use strict';
(() => {
  const $=s=>document.querySelector(s),slides=window.LECTURE,F=window.LectureFigures;
  $('#deck').innerHTML=slides.map((s,i)=>`<section class="slide" id="slide-${i+1}" aria-label="第 ${i+1} 页：${s.title}"><div class="slide-topline">${s.part}</div>${i?`<h2>${s.title}</h2>`:''}${s.body}</section>`).join('');
  let figureNumber=0;
  document.querySelectorAll('figcaption').forEach(el=>{figureNumber++;el.textContent=el.textContent.replace(/^图\s*\d+/,`图 ${figureNumber}`);});
  const mathErrors=[];
  document.querySelectorAll('[data-tex]').forEach(el=>{
    try{window.katex.render(el.dataset.tex,el,{displayMode:el.classList.contains('math'),throwOnError:true,output:'htmlAndMathml',strict:'error',trust:false});}
    catch(e){el.textContent=el.dataset.tex;el.classList.add('math-error');mathErrors.push(e.message);}
  });
  window.lectureMathErrors=mathErrors;
  $('#contents-links').innerHTML=slides.map((s,i)=>`<a href="#/${i+1}"><span>${String(i+1).padStart(2,'0')}</span>${s.title}</a>`).join('');
  let current=0,paused=matchMedia('(prefers-reduced-motion: reduce)').matches,frameId=null,last=0,time=0,ewaldPlay=false,phi=0,printMode=false,runStep=0;
  const draw=()=>{
    const active=printMode?document:$('.slide.active');if(!active)return;
    if(active.querySelector('#bragg'))F.drawBragg($('#bragg'),Number($('#bragg-angle').value),time);
    if(active.querySelector('#ewald-static'))F.drawEwald($('#ewald-static'),0);
    if(active.querySelector('#ewald')){F.drawEwald($('#ewald'),phi);$('#ewald-value').textContent=`${phi.toFixed(1)}°`;$('#ewald-angle').value=String(phi);}
    if(active.querySelector('#overlap'))F.drawOverlap($('#overlap'),Number($('#separation').value)/25);
  };
  const moving=()=>!printMode&&!document.hidden&&!paused&&($('.slide.active #bragg')||($('.slide.active #ewald')&&ewaldPlay));
  function frame(stamp){frameId=null;if(!moving())return;const dt=last?Math.min(70,stamp-last):0;last=stamp;time+=dt;if($('.slide.active #ewald')&&ewaldPlay){phi+=dt*.008;if(phi>35)phi=-35;}draw();frameId=requestAnimationFrame(frame);}
  function animate(){if(frameId!==null)cancelAnimationFrame(frameId);frameId=null;last=0;draw();if(moving())frameId=requestAnimationFrame(frame);}
  function motion(){document.body.classList.toggle('paused',paused);$('#motion').textContent=paused?'播放动画':'暂停动画';$('#motion').setAttribute('aria-pressed',String(paused));animate();}
  function menu(open){$('#contents').hidden=!open;$('#menu-toggle').setAttribute('aria-expanded',String(open));if(open)$('#menu-close').focus();else $('#menu-toggle').focus();}
  function show(i){current=Math.max(0,Math.min(slides.length-1,i));document.querySelectorAll('.slide').forEach((s,j)=>{s.classList.toggle('active',j===current);s.inert=j!==current&&!printMode;});$('#page').textContent=`${current+1} / ${slides.length}`;$('#prev').disabled=current===0;$('#next').disabled=current===slides.length-1;$('#section-name').textContent=slides[current].part;$('#chapter-label').textContent=slides[current].title;$('#notes-text').textContent=slides[current].notes;$('#progress-fill').style.width=`${100*(current+1)/slides.length}%`;document.querySelectorAll('#contents-links a').forEach((a,j)=>{if(j===current)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});document.title=`${current+1}. ${slides[current].title} · Ariadne`;window.scrollTo(0,0);animate();}
  const go=i=>{location.hash=`/${Math.max(0,Math.min(slides.length-1,i))+1}`;};
  const route=()=>{const m=location.hash.match(/^#\/(\d+)$/);show(m?Number(m[1])-1:0);};
  $('#prev').onclick=()=>go(current-1);$('#next').onclick=()=>go(current+1);$('#menu-toggle').onclick=()=>menu($('#contents').hidden);$('#menu-close').onclick=()=>menu(false);
  $('#contents-links').onclick=e=>{if(e.target.closest('a')){menu(false);$('#deck').focus();}};
  $('#notes-toggle').onclick=()=>{$('#notes').hidden=!$('#notes').hidden;$('#notes-toggle').setAttribute('aria-pressed',String(!$('#notes').hidden));};
  $('#motion').onclick=()=>{paused=!paused;motion();};
  $('#fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('#fullscreen').textContent='全屏未获允许';}};
  $('#print').onclick=()=>window.print();
  $('#bragg-angle').oninput=()=>{$('#bragg-value').textContent=`${$('#bragg-angle').value}°`;draw();};
  $('#ewald-angle').oninput=()=>{phi=Number($('#ewald-angle').value);ewaldPlay=false;$('#ewald-play').textContent='播放旋转';animate();};
  $('#ewald-play').onclick=()=>{ewaldPlay=!ewaldPlay;if(ewaldPlay){paused=false;motion();}$('#ewald-play').textContent=ewaldPlay?'停止旋转':'播放旋转';animate();};
  $('#ewald-zero').onclick=()=>{phi=0;ewaldPlay=false;$('#ewald-play').textContent='播放旋转';animate();};
  $('#separation').oninput=()=>{$('#separation-value').textContent=(Number($('#separation').value)/25).toFixed(2);draw();};
  const logs=['流程示意：工具已有输出，尚需检查数值与科学条件。','数值检查通过：优化满足停止条件；检查重叠参数是否可辨识。','可辨识性：示例数据只支持 I₁+I₂，分别强度未获数据支持。','保留新的处理分支与强度组合；科学接受暂缓，继续比较证据。'];
  function run(){document.querySelectorAll('#run-flow .state').forEach((s,i)=>s.classList.toggle('current',i===runStep));$('#run-log').textContent=logs[runStep];$('#run-step').disabled=runStep===3;}
  $('#run-step').onclick=()=>{runStep=Math.min(3,runStep+1);run();};$('#run-reset').onclick=()=>{runStep=0;run();};
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape'){if(!$('#contents').hidden)menu(false);$('#notes').hidden=true;$('#notes-toggle').setAttribute('aria-pressed','false');return;}
    if(!$('#contents').hidden){if(e.key==='Tab'){const nodes=[...$('#contents').querySelectorAll('button,a')],first=nodes[0],last=nodes.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}return;}
    if(e.ctrlKey||e.metaKey||e.altKey||e.target.matches('input,textarea,select'))return;
    if(e.key===' '&&e.target.matches('button,a,summary'))return;
    if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(current+1);}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(current-1);}else if(e.key==='Home')go(0);else if(e.key==='End')go(slides.length-1);else if(e.key.toLowerCase()==='n')$('#notes-toggle').click();else if(e.key.toLowerCase()==='m')menu(true);else if(e.key.toLowerCase()==='f')$('#fullscreen').click();
  });
  window.addEventListener('hashchange',route);window.addEventListener('resize',draw);document.addEventListener('visibilitychange',animate);matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',e=>{paused=e.matches;motion();});
  window.addEventListener('beforeprint',()=>{printMode=true;document.body.classList.add('printing');document.querySelectorAll('.slide').forEach(s=>{s.inert=false;});animate();});window.addEventListener('afterprint',()=>{printMode=false;document.body.classList.remove('printing');show(current);});
  motion();route();
})();
