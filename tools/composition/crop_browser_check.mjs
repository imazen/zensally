import {writeFile} from 'node:fs/promises';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result);}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params);});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value;}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('article').length===56"))break;await new Promise(r=>setTimeout(r,100));}
const result=await evaluate(`(async()=>{
 const failures=[],checks=[];
 for(const r of ['3:1','1:3','6:1','1:6'])for(const m of ['minimal','maximal'])for(const g of [false,true]){
  ratio.value=r;mode.value=m;guard.checked=g;ratio.dispatchEvent(new Event('change'));let contained=0;
  for(const row of data){const s=selection(row);if(!s){contained++;continue}const b=s.box;if(b.some(x=>!Number.isFinite(x))||b[0]<0||b[1]<0||b[2]>1||b[3]>1||b[0]>=b[2]||b[1]>=b[3])failures.push('invalid box');const[x,y]=r.split(':').map(Number);if(Math.abs((b[2]-b[0])*row.width/((b[3]-b[1])*row.height)-x/y)>1e-5)failures.push('ratio mismatch');if(g&&row.heads.some(p=>!(b[0]<=p[0]&&b[1]<=p[1]&&b[2]>=p[2]&&b[3]>=p[3])))failures.push('head guard mismatch');}
  if(document.querySelectorAll('article').length!==56||document.querySelectorAll('article svg').length!==112)failures.push('missing cards');checks.push({ratio:r,mode:m,guard:g,contain:contained});
 }
 const urls=[...new Set([...document.querySelectorAll('svg image')].map(el=>el.getAttribute('href')))];
 for(const url of urls){const image=new Image();image.src=url;try{await image.decode();if(!image.naturalWidth)failures.push('empty image')}catch(e){failures.push('decode: '+url)}}
 const semanticChecks=[];
 if(data.some(row=>row.semantic)){
  document.querySelector('#semantic').checked=true;
  for(const r of ratios)for(const m of ['minimal','maximal']){
   ratio.value=r;mode.value=m;guard.checked=true;ratio.dispatchEvent(new Event('change'));let crops=0;
   for(const row of data.filter(row=>row.semantic)){const s=selection(row);if(!s)continue;crops++;for(const region of row.semantic.regions){const p=region.box_percent.map(v=>v/100),b=s.box;if(!(b[0]<=p[0]&&b[1]<=p[1]&&b[2]>=p[2]&&b[3]>=p[3]))failures.push('semantic containment mismatch')}for(const p of row.heads){const b=s.box;if(!(b[0]<=p[0]&&b[1]<=p[1]&&b[2]>=p[2]&&b[3]>=p[3]))failures.push('old head protection changed')}}
   semanticChecks.push({ratio:r,mode:m,crops});
  }
 }
 ratio.value='1:1';mode.value='maximal';guard.checked=true;ratio.dispatchEvent(new Event('change'));
 return {sources:data.length,images:urls.length,checks,semanticChecks,failures};
})()`);
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');
if(result.sources!==Number(expected)||result.images!==Number(expected)||result.checks.length!==16||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
// Bounded eight-scene screenshot; independent browser display only, never training pixels.
await evaluate(`(()=>{const ids=new Set(['selfie-family-hike','birthday-cake-kitchen','vacation-museum-cafe','linen-shirt-model','leather-tote-model','garden-furniture-context','coffee-bar-context','dog-harness-white']);for(const el of document.querySelectorAll('article'))if(!ids.has(el.querySelector('h2').textContent))el.remove();document.querySelector('main').style.gridTemplateColumns='1fr 1fr';})()`);
await call('Emulation.setDeviceMetricsOverride',{width:1500,height:2200,deviceScaleFactor:1,mobile:false});const dimensions=await evaluate('({width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})');const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:dimensions.width,height:dimensions.height,scale:1}});const bytes=Buffer.from(shot.data,'base64');if(bytes.length>8*1024*1024)throw Error('bounded screenshot exceeds 8 MiB');await writeFile(output.replace('.json','-witness.png'),bytes);ws.close();console.log(JSON.stringify(result));
