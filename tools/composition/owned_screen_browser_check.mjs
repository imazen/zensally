import {writeFile} from 'node:fs/promises';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;
const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('article').length===12"))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{
 const failures=[],checks=[],filterChecks=[];let logicalRequests=0;
 for(const source of data)for(const ratio of ratios)for(const mode of ['minimal','maximal']){
  logicalRequests++;const c=source.tasks[ratio][mode];if(!c)continue;const[l,t,r,b]=c.pixel_box,[p,q]=ratio.split(':').map(Number);
  if(!(0<=l&&l<r&&r<=source.width&&0<=t&&t<b&&b<=source.height)||(r-l)*q!==(b-t)*p)failures.push('invalid integer crop');
  for(const h of source.heads)if(l>Math.floor(h[0]*source.width)||t>Math.floor(h[1]*source.height)||r<Math.ceil(h[2]*source.width)||b<Math.ceil(h[3]*source.height))failures.push('supplied face region cut');
 }
 for(const ratio of ['3:1','1:3','6:1','1:6'])for(const mode of ['minimal','maximal'])for(const overlays of [false,true]){
  document.querySelector('#ratio').value=ratio;document.querySelector('#mode').value=mode;for(const id of ['faces','poses','people'])document.querySelector('#'+id).checked=overlays;page=0;render();
  if(document.querySelectorAll('article').length!==12||document.querySelectorAll('svg').length!==24)failures.push('missing page cards');
  for(const svg of document.querySelectorAll('svg'))if(svg.getAttribute('preserveAspectRatio')!=='xMidYMid meet')failures.push('stretched SVG');
  for(const article of document.querySelectorAll('article'))if(!article.querySelectorAll('svg')[1].querySelector('image').hasAttribute('clip-path'))failures.push('missing crop clip');
  for(const el of document.querySelectorAll('rect,circle'))for(const a of [...el.attributes])if(['x','y','width','height','cx','cy','r'].includes(a.name)&&!Number.isFinite(Number(a.value)))failures.push('invalid overlay');
  checks.push({ratio,mode,overlays});
 }
 for(const filter of ['face','pose','person','witness']){document.querySelector('#filter').value=filter;page=0;render();const count=filtered().length;if(document.querySelectorAll('article').length!==Math.min(12,count))failures.push('filter pagination');filterChecks.push({filter,count})}
 document.querySelector('#filter').value='all';page=0;render();document.querySelector('#next').click();if(document.querySelector('article').dataset.source!==data[12].id)failures.push('next page');document.querySelector('#previous').click();
 document.querySelector('article button').click();if(document.querySelectorAll('button.chosen').length!==1)failures.push('review missing');
 const saved=JSON.parse(localStorage.getItem(storage));if(Object.values(saved).some(r=>r.face_coverage_certified||r.training_admission_automatic||!r.bank_sha256))failures.push('review identity/admission changed');
 let decoded=0;for(let start=0;start<data.length;start+=8)await Promise.all(data.slice(start,start+8).map(async source=>{const image=new Image();image.src=source.image;try{await image.decode();if(!image.naturalWidth)failures.push('empty image');else decoded++}catch(e){failures.push('decode: '+source.id)}}));
 return {sources:data.length,logicalRequests,checks,filterChecks,decoded,failures};
})()`);
result.exceptions=exceptions;
if(result.sources!==Number(expected)||result.decoded!==Number(expected)||result.logicalRequests!==Number(expected)*22||result.checks.length!==16||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
// Two existing photo witnesses only; browser display, never training pixels.
await evaluate(`(()=>{document.querySelector('#filter').value='witness';document.querySelector('#ratio').value='1:3';document.querySelector('#mode').value='maximal';for(const id of ['faces','poses','people'])document.querySelector('#'+id).checked=true;page=0;render();for(const el of document.querySelectorAll('article'))if(!['11048','11214'].includes(el.dataset.source))el.remove();document.querySelector('main').style.gridTemplateColumns='1fr';for(const el of document.querySelectorAll('svg'))el.style.maxHeight='550px'})()`);
await call('Emulation.setDeviceMetricsOverride',{width:1200,height:1800,deviceScaleFactor:1,mobile:false});
// Fit the whole witness before measuring pixels: full-page capture can remove
// the scrollbar and reflow a narrower viewport, shifting gutter coordinates.
const fullHeight=await evaluate('document.documentElement.scrollHeight+32');
await call('Emulation.setDeviceMetricsOverride',{width:1200,height:fullHeight,deviceScaleFactor:1,mobile:false});
result.witness_viewport={width:1200,height:fullHeight,device_scale_factor:1};
result.gutter_samples=await evaluate(`(()=>{window.scrollTo(0,0);const samples=[];for(const article of document.querySelectorAll('article')){const svg=article.querySelectorAll('svg')[1],v=svg.viewBox.baseVal,m=svg.getScreenCTM(),r=svg.getBoundingClientRect(),cy=v.y+v.height/2;for(const[side,cx]of [['left',v.x],['right',v.x+v.width]]){const point=new DOMPoint(cx,cy).matrixTransform(m),x=Math.round(point.x+(side==='left'?-8:8)),y=Math.round(point.y);if(x<r.left+2||x>r.right-2)throw Error('missing tall-crop gutter');samples.push({source:article.dataset.source,side,x,y,expected_rgb:[23,27,34]})}}return samples})()`);
const dimensions=await evaluate('({width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})');
const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,...dimensions,scale:1}});
const bytes=Buffer.from(shot.data,'base64');if(bytes.length>8*1024*1024)throw Error('witness screenshot exceeds8MiB');
result.witness_screenshot=output.replace('.json','-witness-'+Date.now()+'.png');
await writeFile(result.witness_screenshot,bytes,{flag:'wx'});
await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();
console.log(JSON.stringify(result));
