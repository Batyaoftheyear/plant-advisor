"""Read-only audit of the revised SFT files and their cross-file consistency."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'results'/'dataset'
read=lambda path: [json.loads(line) for line in (DATA/path).read_text(encoding='utf-8').splitlines() if line.strip()]
rows=read('sft_metadata.jsonl'); orig=read('sft_metadata_raw.jsonl')
train=read('train.jsonl'); val=read('validation.jsonl'); questions=read('eval_questions.jsonl')
assert len(rows)==len(orig)==400 and len(train)==360 and len(val)==40 and len(questions)==30
assert all(x['messages'][0]['role']=='user' and x['messages'][1]['role']=='assistant' for x in train+val)
by_id={r['id']:r for r in rows}
for s,examples in [('train',train),('validation',val)]:
 assert [(r['question'],r['answer']) for r in rows if r['split']==s]==[(x['messages'][0]['content'],x['messages'][1]['content']) for x in examples]
assert len(set(r['question'] for r in rows))==400
assert len(set(r['answer'] for r in rows))==400
assert not any('пустынного кактуса' in r['question'] and re.search(r'лист[а-я]*',r['question']) for r in rows)
assert not any(re.search(r'(\bна\s+(?:хлорофитуму|фаленопсису|фикусу|драцены)|что изменить в содержании)',r['question'], re.I) for r in rows)

allq=[r['question'] for r in rows]; evalq=[r['question'] for r in questions]
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5))
xx=vec.fit_transform(allq+evalq); sims=(xx[:400]@xx[400:].T).toarray()
idx=sims.argmax(axis=1); mx=sims.max(axis=1)
with (OUT/'eval_similarity_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['id','split','sft_question','closest_eval_id','closest_eval_question','cosine_similarity'])
 for k in sorted(range(len(rows)),key=lambda k: -mx[k]):
  w.writerow([rows[k]['id'],rows[k]['split'],allq[k],questions[int(idx[k])]['id'],evalq[int(idx[k])],round(float(mx[k]),5)])

# More meaningful cross-split question similarity is retained as an audit rather than a leakage declaration.
xt=vec.fit_transform([r['question'] for r in rows if r['split']=='train']+[r['question'] for r in rows if r['split']=='validation'])
inter=(xt[:360]@xt[360:].T).toarray()

changed=[r for r in rows if r['revision_status']=='revised']
reason_counts=Counter(t.split(':')[0] for r in changed for t in r['revision_reasons'])
used_sources=sorted(set(s for r in rows for s in r['source_ids']))

summary={
 'record_count':len(rows),'train_count':len(train),'validation_count':len(val),
 'task_counts': dict(Counter(r['task'] for r in rows)),
 'plant_count':len(set(r['plant'] for r in rows)),
 'unique_questions':len(set(allq)),
 'unique_answers':len(set(r['answer'] for r in rows)),
 'original_duplicate_answers':len(orig)-len({r['answer'] for r in orig}),
 'revised_duplicate_answers':0,
 'revised_count':len(changed),'unchanged_count':400-len(changed),
 'change_categories':dict(reason_counts),
 'exact_eval_question_duplicates':0,
 'eval_tf_idf_max':float(mx.max()),'eval_tf_idf_ge_0.85':int((mx>=0.85).sum()),
 'cross_split_question_tf_idf_max':float(inter.max()),'cross_split_pairs_ge_0.85':int((inter>=0.85).sum()),
 'used_source_ids':used_sources,
 'source_urls_verified_as_general_references':False,
 'manual_expert_review_complete':False,
 'note':'TF-IDF controls text similarity, not conceptual leakage. A source URL is general supporting context, not verification of every claim.'
}
(OUT/'dataset_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('\nTop eval similarities:')
for k in mx.argsort()[-12:][::-1]:
 print(rows[k]['id'], round(float(mx[k]),3), rows[k]['question'], ' || ', evalq[int(idx[k])])
