"""Observe public Instagram account grids through Aside; never infer a trend from recency alone.

Run: python3 collect/ig_trends.py --max-posts 4
Each invocation creates a timestamped observation. No scheduler is installed.
Requires the Aside browser profile used for this project. Read `aside guide` before use.
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    {'account': 'readwhat.memes', 'seed': 'https://www.instagram.com/p/DflkpvIgTm3/'},
    {'account': 'foodie.at_talk', 'seed': 'https://www.instagram.com/p/DfjpEDrDSUi/'},
    {'account': 'cysh_meme', 'seed': 'https://www.instagram.com/p/Dfj-T06p5Uo/'},
    {'account': 'meme._.maomao', 'seed': 'https://www.instagram.com/p/Dflod7eEtYO/'},
    {'account': 'ggyoung5278', 'seed': 'https://www.instagram.com/p/DflxmZOvutV/'},
]

JS = r'''
var cfg = __CONFIG__;
var opened = await listBrowserTabs();
var existing = opened.find(t => t.url === cfg.seed);
var owns = !existing;
var p = existing ? await attachBrowserTab(existing.targetId) : await openTab(cfg.seed);
var result = {account: cfg.account, started_at: new Date().toISOString(), posts: [], errors: []};
try {
  var seedSnapshot = await snapshot(p);
  var profileHref = await p.getByRole('link',{name:cfg.account,exact:true}).first().getAttribute('href');
  var profileUrl = new URL(profileHref,p.url());
  if(profileUrl.origin !== 'https://www.instagram.com' || profileUrl.pathname !== '/'+cfg.account+'/') throw new Error('Unexpected profile link');
  await p.goto(profileUrl.href);
  var grid = await snapshot(p);
  var gridClose = grid.tree.match(/- dialog:\n\s+- button \[ref=(\w+)\][^\n]*:\n\s+- img "닫기"/);
  if(gridClose) {await p.locator(gridClose[1]).click();grid=await snapshot(p);}
  result.profile_url = p.url();
  result.profile_snapshot = grid.tree;
  var refs = [...grid.tree.matchAll(/- link(?: "[^"\n]*")? \[ref=(\w+)\]/g)].map(m=>m[1]);
  var links = [];
  for(var ref of refs) {
    var href = await p.locator(ref).getAttribute('href');
    if(!href) continue;
    var u = new URL(href,p.url());
    if(u.origin !== 'https://www.instagram.com') continue;
    var match = u.pathname.match(/^\/([^/]+)\/(p|reel)\/([^/]+)\/$/);
    if(match && match[1] === cfg.account && !links.some(x=>x.id===match[3])) {
      var afterRef = grid.tree.split('[ref='+ref+']')[1] || '';
      var block = afterRef.split('\n    - link')[0];
      var dateHint = block.match(/(?:Photo|Video) by .*? on ([A-Z][a-z]+ \d{2}, \d{4})\./);
      links.push({url:u.href,id:match[3],position:links.length+1,date_hint:dateHint ? dateHint[1] : null});
    }
  }
  result.grid_links = links;
  if(!links.length) result.errors.push('No public grid post links; login wall or UI changed');
  // A pinned post can precede this week's posts. Use date hints only when all
  // grid entries have them; otherwise preserve order and flag unknown age later.
  var ordered = [...links];
  result.grid_order_basis = links.length && links.every(x=>x.date_hint) ? 'media_alt_date_descending' : 'grid_order_age_checked_after_open';
  if(result.grid_order_basis === 'media_alt_date_descending') ordered.sort((a,b)=>Date.parse(b.date_hint+' UTC')-Date.parse(a.date_hint+' UTC') || a.position-b.position);
  for(var link of ordered.slice(0,cfg.maxPosts)) {
    try {
      await p.goto(link.url);
      var s = await snapshot(p);
      var close = s.tree.match(/- dialog:\n\s+- button \[ref=(\w+)\][^\n]*:\n\s+- img "닫기"/);
      if(close) {await p.locator(close[1]).click();s=await snapshot(p);}
      result.posts.push({id:link.id,url:p.url(),grid_position:link.position,observed_at:new Date().toISOString(),snapshot:s.tree,body:await p.getByRole('main').innerText()});
    } catch(err) {result.errors.push({url:link.url,error:String(err)});}
  }
} catch(err) {result.errors.push(String(err));}
finally {if(owns) await closeTab(p);}
console.log(JSON.stringify(result));
'''

def run_source(source, limit, profile):
    script = JS.replace('__CONFIG__', json.dumps({**source,'maxPosts':limit},ensure_ascii=False))
    # stdin and argument arrays avoid shell interpolation of any page content.
    command = ['aside','repl','--json','--file','-']
    if profile: command += ['--profile',profile]
    proc = subprocess.run(command,input=script,text=True,capture_output=True,timeout=150)
    if proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout)
    envelope=json.loads(proc.stdout)
    run=envelope.get('run',{})
    if run.get('error'): raise RuntimeError(str(run['error']))
    for item in reversed(run.get('content',[])):
        if item.get('type') != 'text': continue
        for line in reversed(item['text'].splitlines()):
            if line.startswith('{'):
                try:
                    obj=json.loads(line)
                    if 'posts' in obj and 'account' in obj: return obj
                except json.JSONDecodeError: pass
    raise RuntimeError('No structured observation returned by Aside')

def metrics(snapshot):
    # Only the post-action region: don't confuse comment likes with post likes.
    regions=re.findall(r'    - region:\n(.*?)(?=\n    - |\Z)',snapshot,re.S)
    for region in regions:
        if 'img "좋아요"' not in region: continue
        parsed={}
        for key,icon in [('likes','좋아요'),('comments','댓글 달기')]:
            match=re.search('img "'+icon+r'"\n\s+- button "([\d.,]+(?:천|만|[KkMm])?)"',region)
            label=match[1] if match else None
            parsed[key]={'label':label,'exact_value':int(label.replace(',','')) if label and re.fullmatch(r'[\d,]+',label) else None}
        return parsed
    return {key:{'label':None,'exact_value':None} for key in ['likes','comments']}

def annotate(post,account):
    post=dict(post,source_account=account)
    tail=post['body'].split('좋아요 또는 댓글을 남기려면',1)[0]
    ages=re.findall(r'(?m)^(\d+)\s*(분|시간|일) 전$',tail)
    post['post_age_label']=ages[-1][0]+ages[-1][1]+' 전' if ages else None
    if ages:
        n,unit=ages[-1]; unit_hours={'분':1/60,'시간':1,'일':24}[unit]
        lower=int(n)*unit_hours; upper=lower+unit_hours
        post['post_age_hours_interval']=[lower,upper]
        post['within_72h']= upper<=72
    else:
        weeks=re.search(r'(?m)^(\d+)주$',tail)
        post['post_age_label']=weeks[0] if weeks else None
        post['post_age_hours_interval']=[int(weeks[1])*168,(int(weeks[1])+1)*168] if weeks else None
        post['within_72h']=False if weeks else None
    post['metrics']=metrics(post['snapshot'])
    post['content_hash']=hashlib.sha256(post['body'].encode()).hexdigest()
    post['coverage']='public_grid_prefix_and_visible_comments; not complete feed'
    post['novelty_status']='unreviewed; recent publication does not establish new meme'
    return post

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-posts',type=int,default=4)
    ap.add_argument('--profile',default='46cc2775-95ce-4811-97db-62e27cd49128')
    ap.add_argument('--sources',type=Path,help='JSON list of account and previously observed seed URL')
    ap.add_argument('--output',type=Path,default=ROOT/'data'/'trends'/'instagram')
    args=ap.parse_args()
    if not 1<=args.max_posts<=12: ap.error('--max-posts must be between 1 and 12')
    sources=json.loads(args.sources.read_text()) if args.sources else DEFAULT_SOURCES
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder=args.output/'runs'/stamp; folder.mkdir(parents=True,exist_ok=False)
    state_path=args.output/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'posts':{},'runs':[]}
    summary={'run_id':stamp,'previous_runs':len(state['runs']),'sources':[],'observed_posts':0,'newly_seen_posts':0,'recent_72h_posts':0,
      'trend_status':'insufficient_baseline','note':'Newly seen means new to this collector, not a new meme. No trend rank is computed.'}
    for source in sources:
        print('Observing '+source['account'],flush=True)
        try:
            observation=run_source(source,args.max_posts,args.profile)
            observation['posts']=[annotate(p,source['account']) for p in observation['posts']]
            for post in observation['posts']:
                previous=state['posts'].get(post['id'])
                post['first_seen_at']=previous['first_seen_at'] if previous else post['observed_at']
                post['previous_observation']=previous
                post['metric_delta']={}
                for key in ['likes','comments']:
                    old=(previous or {}).get('metrics',{}).get(key,{}).get('exact_value')
                    new=post['metrics'][key]['exact_value']
                    post['metric_delta'][key]=new-old if old is not None and new is not None else None
                if not previous: summary['newly_seen_posts']+=1
                state['posts'][post['id']]={k:post[k] for k in ['first_seen_at','observed_at','url','metrics','content_hash']}
                summary['recent_72h_posts']+=post['within_72h'] is True
            (folder/(source['account']+'.json')).write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
            summary['observed_posts']+=len(observation['posts'])
            summary['sources'].append({'account':source['account'],'count':len(observation['posts']),'errors':observation['errors']})
        except Exception as exc:
            summary['sources'].append({'account':source['account'],'count':0,'errors':[str(exc)]})
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    state['runs'].append({'id':stamp,'observed_posts':summary['observed_posts']})
    temporary=state_path.with_suffix('.tmp'); temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temporary.replace(state_path)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Saved '+str(folder),flush=True)
    if summary['observed_posts']==0: raise SystemExit(1)

if __name__=='__main__': main()
