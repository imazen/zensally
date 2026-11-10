// Read-only exact-name Google checks. A challenge is never recorded as no matches.
import {writeFile} from 'node:fs/promises';
const [endpoint,output,...names]=process.argv.slice(2);
const pages=await(await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let serial=0;const pending=new Map();
ws.addEventListener('message',e=>{const v=JSON.parse(e.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
await call('Page.enable');
const results=[];
for(const name of names){
 const url='https://www.google.com/search?q='+encodeURIComponent('"'+name+'"');
 await call('Page.navigate',{url});
 let text='';
 for(let i=0;i<60;i++){
  await new Promise(r=>setTimeout(r,250));
  const x=await call('Runtime.evaluate',{expression:'document.body?.innerText',returnByValue:true});text=x.result.value||'';
  if(text.includes('did not match any documents')||text.includes('No results found')||text.includes('Our systems have detected')||text.includes('Before you continue'))break;
 }
 const dom=await call('Runtime.evaluate',{expression:'document.documentElement.outerHTML',returnByValue:true});
 const links=await call('Runtime.evaluate',{expression:'JSON.stringify([...document.links].map(a=>({text:a.innerText,href:a.href})))',returnByValue:true});
 await writeFile(output+'/'+name.toLowerCase()+'-google-dom.html',dom.result.value||'');
 results.push({name,url,text,links:JSON.parse(links.result.value),no_exact_matches:text.includes('did not match any documents')||text.includes('No results found')});
 await writeFile(output+'/brand-google-browser.json',JSON.stringify(results,null,2));
 console.log(name,text.slice(-1800));
}
ws.close();
