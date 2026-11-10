import {writeFile,mkdir} from 'node:fs/promises';
import {dirname,join} from 'node:path';
const[endpoint,url,expected,output]=process.argv.slice(2);
const pages=await(await fetch(endpoint+'/json/list')).json(),ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1500,height:1600,deviceScaleFactor:1,mobile:false});await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate('document.querySelectorAll("section").length===5'))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{const failures=[],records=await(await fetch('readbacks.json')).json();for(const img of document.images){await img.decode();const path=new URL(img.src).pathname.replace(/^\\//,'');const row=records.find(r=>r.file===path);if(!row||img.naturalWidth!==row.size[0]||img.naturalHeight!==row.size[1])failures.push('decoded pixel dimensions differ');if(getComputedStyle(img).objectFit!=='contain')failures.push('stretched image')}return{sources:document.querySelectorAll('section').length,images:document.images.length,crops:document.querySelectorAll('figure').length,rejections:document.querySelectorAll('figure.bad').length,failures}})()`);
if(result.sources!==Number(expected)||result.images!==43||result.crops!==38||result.rejections!==1||result.failures.length||exceptions.length)throw Error(JSON.stringify({...result,exceptions}));
const directory=join(dirname(output),'screenshots');await mkdir(directory);result.screenshot_bytes=0;result.screenshots=[];
for(let i=0;i<result.sources;i++){const rect=await evaluate(`(()=>{const r=document.querySelectorAll('section')[${i}].getBoundingClientRect();return{x:r.x,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()`);const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:rect}),bytes=Buffer.from(shot.data,'base64');result.screenshot_bytes+=bytes.length;if(result.screenshot_bytes>32*1024**2)throw Error('bounded screenshot budget exceeded');const name='subject-'+i+'.png';await writeFile(join(directory,name),bytes);result.screenshots.push(name)}
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();console.log(JSON.stringify(result));

