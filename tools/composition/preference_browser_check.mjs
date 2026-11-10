import {writeFile} from 'node:fs/promises';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('article').length===5"))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{
 const failures=[],checks=[],urls=new Set();
 for(const ratio of ['3:1','1:3','6:1','1:6'])for(const mode of ['minimal','maximal'])for(const raw of [false,true]){
  document.querySelector('#ratio').value=ratio;document.querySelector('#mode').value=mode;document.querySelector('#raw').checked=raw;render();
  if(document.querySelectorAll('article').length!==5||document.querySelectorAll('.card').length!==25)failures.push('missing comparison cards');
  for(const [i,article] of [...document.querySelectorAll('article')].entries()){
   const source=data.sources[i],task=source.tasks.find(t=>t.ratio===ratio&&t.mode===mode),images=[...article.querySelectorAll('img')];
   await Promise.all(images.map(async image=>{urls.add(image.src);try{await image.decode();const r=image.getBoundingClientRect();if(!image.naturalWidth||Math.abs(r.width/r.height-image.naturalWidth/image.naturalHeight)>.01)failures.push('decode or stretched pixels')}catch(e){failures.push('decode: '+image.src)}}));
   for(const [j,choice] of task.choices.entries()){const expected=choice.action==='crop'||raw?choice.image:source.image;if(images[j+1].src!==expected)failures.push('contain/rank display mismatch')}
  }
  checks.push({ratio,mode,raw});
 }
 document.querySelector('#labels').checked=true;render();
 if(!document.body.textContent.includes('Weak teacher:'))failures.push('label toggle missing');
 document.querySelector('article button').click();
 if(document.querySelectorAll('button.chosen').length!==1||!document.querySelector('#count').textContent.startsWith('1 '))failures.push('review did not persist');
 const saved=JSON.parse(localStorage.getItem('zensally:'+data.study));
 if(Object.values(saved).some(r=>r.face_coverage_certified||r.training_admission_automatic))failures.push('review changed admission');
 return {sources:data.sources.length,checks,unique_images:urls.size,failures};
})()`);
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();
if(result.sources!==Number(expected)||result.checks.length!==16||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
console.log(JSON.stringify(result));
