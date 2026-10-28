import { readFileSync, openSync, writeSync, fsyncSync, closeSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

// One chronological episode against an already-started local server. No retries,
// store manipulation, warmup prompts, tool execution, or server lifecycle changes.
const [clientPath, inputPath, episode, baseUrl, outputPath, receiptPath] = process.argv.slice(2);
if (!receiptPath) throw Error('Usage: node replay-timed.mjs CLIENT INPUT.jsonl EPISODE http://127.0.0.1:PORT OUT.jsonl RUN-RECEIPT.json');
const url = new URL(baseUrl);
if (url.hostname !== '127.0.0.1' || url.protocol !== 'http:') throw Error('Local loopback HTTP only');
const sha = x => createHash('sha256').update(x).digest('hex');
const bytes = readFileSync(inputPath);
const assigned = bytes.toString('utf8').trim().split('\n').map(JSON.parse).filter(x => x.episode_id === episode);
if (!assigned.length || new Set(assigned.map(x => x.id)).size !== assigned.length) throw Error('Empty episode or duplicate request IDs');
const receipt = JSON.parse(readFileSync(receiptPath));
for (const key of ['arm','source_commit','binary_sha256','model_identity','server_argv','store_initial_state','power','thermal','fifo_owner']) {
  if (receipt[key] === undefined || receipt[key] === '') throw Error(`Missing run receipt: ${key}`);
}
if (!['main','history-off','history-on','smoke'].includes(receipt.arm)) throw Error('Unknown arm');
if (receipt.dataset_sha256 !== sha(bytes)) throw Error('Dataset hash differs from receipt');
const provenance = JSON.parse(readFileSync(clientPath + '.provenance.json'));
if (sha(readFileSync(clientPath)) !== provenance.adapted_sha256) throw Error('Timing client changed');
const { EngineClient, noDeadlineDispatcher } = await import(pathToFileURL(resolve(clientPath)).href);
const client = new EngineClient({baseUrl, timeoutMs:600000});
const fd = openSync(outputPath, 'wx', 0o600);
const emit = row => { writeSync(fd, JSON.stringify(row)+'\n'); fsyncSync(fd); };
let failed = false;
try {
  emit({kind:'assignment',episode,dataset_sha256:sha(bytes),ids:assigned.map(x=>x.id),receipt,timing:provenance,request_timeout_ms:600000});
  for (const item of assigned) {
    if (failed) { emit({kind:'request',id:item.id,status:'unattempted-after-failure'}); continue; }
    const body = {...item.body, stream:true, stream_options:{include_usage:true}};
    emit({kind:'request-start',id:item.id,request_sha256:sha(JSON.stringify(body)),started_at:new Date().toISOString()});
    try {
      const timed = await client.streamTimed('/v1/chat/completions', body);
      let content='', reasoning='', finish=null, usage=null, done=false, first=null;
      const tools=new Map();
      const problems=[];
      for (let i=0;i<timed.frames.length;i++) {
        const frame=timed.frames[i];
        if (frame.data === '[DONE]') { done=true; continue; }
        if (!frame.data) continue;
        let event;
        try { event=JSON.parse(frame.data); } catch { problems.push('invalid-json-frame'); continue; }
        if (event.error) problems.push('server-error-event');
        if (event.usage) usage=event.usage;
        for (const choice of event.choices ?? []) {
          if (choice.index !== 0) { problems.push('unexpected-choice'); continue; }
          const d=choice.delta ?? {};
          if ((d.content || d.reasoning_content || d.reasoning || d.tool_calls?.length) && first===null) first=timed.frameTimesMs[i];
          content+=d.content ?? '';
          reasoning+=d.reasoning_content ?? d.reasoning ?? '';
          for (const part of d.tool_calls ?? []) {
            const t=tools.get(part.index) ?? {id:'',type:'function',function:{name:'',arguments:''}};
            t.id+=part.id ?? ''; t.function.name+=part.function?.name ?? ''; t.function.arguments+=part.function?.arguments ?? '';
            tools.set(part.index,t);
          }
          if (choice.finish_reason !== null && choice.finish_reason !== undefined) finish=choice.finish_reason;
        }
      }
      const tool_calls=[...tools.entries()].sort((a,b)=>a[0]-b[0]).map(x=>x[1]);
      for (const t of tool_calls) {
        if (!t.id || !body.tools?.some(x=>x.function?.name===t.function.name)) problems.push('invalid-tool-identity');
        try { JSON.parse(t.function.arguments); } catch { problems.push('invalid-tool-arguments-json'); }
      }
      if (timed.status!==200) problems.push('http-status');
      if (!done || finish===null) problems.push('incomplete-stream');
      if (!Number.isInteger(usage?.completion_tokens) || !Number.isInteger(usage?.prompt_tokens)) problems.push('missing-token-usage');
      if (finish==='tool_calls' && !tool_calls.length) problems.push('missing-tool-call');
      const result={content,reasoning,tool_calls,finish};
      const comparison={...result,tool_calls:tool_calls.map(({function:fn,type})=>({type,function:fn}))};
      failed=problems.length>0;
      emit({kind:'request',id:item.id,status:failed?'failed':'completed',problems,request_sha256:sha(JSON.stringify(body)),http_status:timed.status,first_delta_ms:first===null?null:first-timed.startMs,total_ms:timed.endMs-timed.startMs,usage,finish_reason:finish,output_capped:finish==='length',output_sha256:sha(JSON.stringify(result)),comparison_sha256:sha(JSON.stringify(comparison)),result,timing:timed});
    } catch(error) { failed=true; emit({kind:'request',id:item.id,status:'failed',error:String(error),request_sha256:sha(JSON.stringify(body))}); }
  }
} finally { closeSync(fd); await noDeadlineDispatcher.close(); }
if (failed) process.exitCode=1;
