import {writeFile,mkdir} from 'node:fs/promises';
import {dirname,join} from 'node:path';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('section').length>0"))break;await new Promise(resolve=>setTimeout(resolve,100))}
const result=await evaluate(`(async()=>{
 const failures=[],sections=[...document.querySelectorAll('section')],images=sections.map(s=>s.querySelector('img'));
 for(const image of images){image.loading='eager';try{await image.decode()}catch(error){failures.push('native PNG decode failed: '+image.src)}}
 for(const section of sections){const image=section.querySelector('img');if(image.naturalWidth!==Number(section.dataset.width)||image.naturalHeight!==Number(section.dataset.height))failures.push('native dimensions differ: '+section.id);if(getComputedStyle(image).objectFit!=='contain')failures.push('source rendering may stretch: '+section.id);if(!section.querySelector('details pre').textContent.includes('photorealistic'))failures.push('prompt missing: '+section.id)}
 const aspect=document.querySelector('#aspect');let previews=0;
 for(const ratio of [...aspect.options].filter(o=>o.value!=='source').map(o=>o.value)){
  aspect.value=ratio;aspect.dispatchEvent(new Event('change'));await new Promise(requestAnimationFrame);
  for(const section of sections){const box=section.querySelector('figure').getBoundingClientRect();if(Math.abs(box.width/box.height-Number(ratio))>.01)failures.push('wrong preview ratio: '+section.id+' '+ratio);if(getComputedStyle(section.querySelector('img')).objectFit!=='cover')failures.push('wrong preview rendering');previews++}
  if(!document.querySelector('#warning').textContent.includes('Not a model recommendation'))failures.push('baseline not labelled');
 }
 aspect.value='source';aspect.dispatchEvent(new Event('change'));
 return {sources:sections.length,source_checks:images.length,preview_checks:previews,failures};
})()`);
await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
result.mobileOverflow=await evaluate('document.documentElement.scrollWidth>innerWidth+2');
result.exceptions=exceptions;
if(process.argv[6]==='capture'){
 await call('Emulation.setDeviceMetricsOverride',{width:1600,height:1800,deviceScaleFactor:1,mobile:false});
 await evaluate(`(()=>{const style=document.createElement('style');style.textContent='header,.controls,section p,section details{display:none!important}#cards{grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}section{padding:4px;border-radius:0}section h2{font-size:14px;margin:2px}section figure,section img{height:390px!important;max-height:390px!important}body{margin:8px}main{max-width:none}';document.head.append(style)})()`);
 const directory=join(dirname(output),'source-screenshots');await mkdir(directory);result.screenshots=[];result.screenshot_bytes=0;
 for(let start=0;start<Number(expected);start+=8){
  const clip=await evaluate(`(()=>{const s=[...document.querySelectorAll('section')],a=s[${start}].getBoundingClientRect(),b=s[Math.min(s.length-1,${start+7})].getBoundingClientRect();return{x:0,y:a.y+scrollY,width:1600,height:b.bottom-a.top,scale:1}})()`);
  const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip}),bytes=Buffer.from(shot.data,'base64');result.screenshot_bytes+=bytes.length;
  if(result.screenshot_bytes>32*1024**2)throw Error('source review screenshot budget exceeded');
  const name='sources-'+start+'.png';await writeFile(join(directory,name),bytes);result.screenshots.push(name)
 }
}
await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();
if(result.sources!==Number(expected)||result.source_checks!==Number(expected)||result.preview_checks!==Number(expected)*7||result.failures.length||result.mobileOverflow||exceptions.length)throw Error(JSON.stringify(result));
console.log(JSON.stringify(result));
