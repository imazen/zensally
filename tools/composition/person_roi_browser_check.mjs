import {writeFile} from 'node:fs/promises';
const[endpoint,url,expected,output]=process.argv.slice(2);
const pages=await(await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;
const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('article').length===8"))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{const failures=[],checks=[];for(const source of ['11048','11214'])for(const model of ['all','float16-blaze_face_full_range_sparse','float16-blaze_face_short_range','float16-pose_landmarker_lite','none']){document.querySelector('#source').value=source;document.querySelector('#model').value=model;render();if(document.querySelectorAll('article').length!==(source==='11048'?8:10))failures.push('view count');for(const svg of document.querySelectorAll('svg')){if(svg.getAttribute('preserveAspectRatio')!=='xMidYMid meet')failures.push('stretched');for(const shape of svg.querySelectorAll('rect,circle'))for(const attr of shape.attributes)if(['x','y','width','height','cx','cy','r'].includes(attr.name)&&!Number.isFinite(Number(attr.value)))failures.push('nonfinite')}checks.push({source,model})}let decoded=0;for(const row of data){const image=new Image();image.src=row.image;await image.decode();if(image.naturalWidth!==row.width||image.naturalHeight!==row.height)failures.push('image shape');decoded++}return{views:data.length,decoded,checks,failures}})()`);
result.exceptions=exceptions;
if(result.views!==Number(expected)||result.decoded!==18||result.checks.length!==10||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
await evaluate("document.querySelector('#source').value='11048';document.querySelector('#model').value='all';render();for(const el of document.querySelectorAll('article'))if(el.dataset.view!=='11048-1-person')el.remove();document.querySelector('main').style.gridTemplateColumns='320px'");
await call('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false});
const shot=await call('Page.captureScreenshot',{format:'png'});const bytes=Buffer.from(shot.data,'base64');if(bytes.length>8*1024*1024)throw Error('screenshot budget exceeded');
result.witness_screenshot=output.replace('.json','-witness.png');await writeFile(result.witness_screenshot,bytes,{flag:'wx'});
await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();console.log(JSON.stringify(result));
