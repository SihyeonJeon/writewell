"""Build a portable, human-readable reaction library; does not enable chat routing."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = html.escape

cards = [
    {
        "id": "ven-again", "title": "我再 ven 一次", "meaning_ko": "다시 한 번 물을게",
        "format": "image_and_text", "tags": ["재촉", "재질문", "약속", "여행", "참여자 모집", "催促", "再問一次", "約旅行"],
        "tone": "친한 사이에서 살짝 답답한 척하며 장난스럽게 재촉",
        "image_files": ["images/ven-travel.jpg"],
        "image_text": "我再ven一次\n今年說好的旅行 還去不去？",
        "image_text_method": "assistant_visual_transcription",
        "source_context_ko": "中租租車의 여행 약속 재촉 광고. 같은 표현이 음식점 행사 참여자 모집과 개인 게시물 댓글에서도 확인됐다. ven을 問 자리에 쓴 표기. 최초 유래·발음 기원은 미확인.",
        "when_to_use": ["계속 말만 나오는 여행 계획을 다시 꺼낼 때", "앞서 물었던 가벼운 질문에 답이 없어 재질문할 때", "함께할 사람을 장난스럽게 한 번 더 모집할 때"],
        "image_fit": "이 파일에는 ‘올해 약속한 여행, 갈 거야 말 거야?’가 박혀 있다. 여행 약속 맥락에 사용. 메뉴 선택·참여자 모집에는 텍스트 변형만 사용.",
        "how_to_use": "사진 한 장만 보내거나 짧은 재질문을 붙인다. 텍스트로는 ‘我再 ven 一次，{질문}’ 형태로 쓴다.",
        "examples": [
            {"incoming_ko": "우리 올해 대만 가자고 한 거 또 미뤄지는 거야?", "reply_zh_tw": "我再 ven 一次，今年說好的旅行還去不去？", "reply_ko": "다시 한번 물을게. 올해 가기로 한 여행, 갈 거야 말 거야?", "send_image": True},
            {"incoming_ko": "주말에 훠궈 먹을 사람 물었는데 아무도 답을 안 함", "reply_zh_tw": "我再 ven 一次，週末有人要吃火鍋嗎？", "reply_ko": "다시 한번 물을게. 주말에 훠궈 먹을 사람?", "send_image": False}
        ],
        "avoid_when": ["상대가 이미 명확히 거절한 일을 다시 압박할 때", "진지한 항의·공식 업무 독촉"],
        "source_urls": ["https://www.instagram.com/p/DfWxGFGmbQT/", "https://www.instagram.com/p/DfUNM8TEzo1/", "https://www.instagram.com/p/DfAVdOcnYnI/"],
        "credit": "수집 이미지 게시자: chailease_auto_rental / 中租租車. 밈 표현 최초 창작자는 미확인.",
        "evidence_status": "여러 독립 게시물·댓글의 용법 확인. 최신 급상승 여부는 미검증.",
        "asset_status": "게시물에 실린 이미지 파일을 수정 없이 다운로드. 브랜드 광고 변형본이며 밈 최초 원형은 아님.",
        "download_keys": ["ven"]
    },
    {
        "id": "peacefully-sleepless", "title": "安心的睡不著覺", "meaning_ko": "안심하고 잠을 못 잔다",
        "format": "image_and_text", "tags": ["불안", "반어", "자조", "주식", "투자", "걱정", "睡不著", "反諷", "股票"],
        "tone": "괜찮은 척하지만 전혀 안 괜찮다는 자조",
        "image_files": ["images/stock-sleepless.jpg"],
        "image_text": "說真的，投資本來就要承擔風險\n像我就買龍頭股\n晚上都安心的睡不著覺\n（開玩笑勿認真😂\n不要慌，買到龍頭了",
        "image_text_method": "assistant_visual_transcription; 주요 농담만, 차트 수치 제외",
        "source_context_ko": "대장주(龍頭股)를 사서 안심한다는 말과 불면을 붙인 모순 농담. 차트에 용 머리를 겹쳐 龍頭를 시각적으로 말장난한다.",
        "when_to_use": ["내 투자나 선택을 두고 괜찮은 척할 때", "긴장되는 결과를 기다리며 가볍게 자조할 때"],
        "image_fit": "주식 차트가 포함돼 있어 이미지는 주식·투자 농담에 가장 적합. 면접·발표 대기는 문구만 전용 가능.",
        "how_to_use": "내 상태를 자조하는 말로 쓴다. 상대의 손실을 평가하는 말로 쓰지 않는다. 원문 표기 的를 보존한다.",
        "examples": [
            {"incoming_ko": "대장주 샀으니까 이제 마음 편하지?", "reply_zh_tw": "對啊，安心的睡不著覺。", "reply_ko": "그럼, 안심하고 잠을 못 자지.", "send_image": True},
            {"incoming_ko": "발표 준비 다 했어?", "reply_zh_tw": "準備好了，現在安心的睡不著覺。", "reply_ko": "준비 끝났어. 이제 안심하고 잠을 못 자는 중.", "send_image": False}
        ],
        "avoid_when": ["상대가 실제 큰 손실이나 심한 불면을 진지하게 호소할 때"],
        "source_urls": ["https://www.instagram.com/p/DfEreWik8cE/"],
        "credit": "수집 게시자: tommytsai824. 그림 최초 제작자는 미확인.",
        "evidence_status": "이미지의 농담 구조 확인. 독립 반복·바이럴 규모 미검증.",
        "asset_status": "게시 이미지 파일 그대로 보존. 파일 자체에 원 게시물 캡처와 차트가 포함됨.",
        "download_keys": ["stock"]
    },
    {
        "id": "thanks-boss", "title": "這樣我休兩天耶，謝謝老闆", "meaning_ko": "그럼 이틀 쉬네요. 감사합니다 사장님",
        "format": "image_and_text", "tags": ["출근", "휴일", "억지 긍정", "눈치 없는 척", "회사", "장난", "休假", "謝謝老闆"],
        "tone": "안 좋은 말을 좋은 쪽으로 알아들은 척하는 능청",
        "image_files": ["images/thanks-boss.jpg"],
        "image_text": "老闆：你明天不用來了\n那後天見囉\n老闆：後天也不需要來了\n這樣我休兩天耶\n謝謝老闆",
        "image_text_method": "assistant_visual_transcription; 대화 문구, 시각·워터마크 제외",
        "source_context_ko": "‘내일부터 나오지 마’라는 해고 암시를 휴가로 받아들이는 채팅 농담.",
        "when_to_use": ["쉬고 싶다는 직장인 잡담", "취소된 일정에서 좋은 점을 억지로 찾는 농담", "상대가 이미 농담 중이고 능청스러운 반응을 기대할 때"],
        "image_fit": "직장·출근·휴일 이야기에서 사용. 채팅 이미지 자체에 완결된 농담이 있어 사진만 보내도 됨.",
        "how_to_use": "‘謝謝老闆’처럼 짧게 받는다. 실제 인사 통보에 대한 해석이나 조언으로 쓰지 않는다.",
        "examples": [
            {"incoming_ko": "내일도 모레도 출근 안 하고 싶다", "reply_zh_tw": "這樣我休兩天耶，謝謝老闆。", "reply_ko": "그럼 이틀 쉬네요. 감사합니다 사장님.", "send_image": True},
            {"incoming_ko": "오늘 약속 취소됐어. 갑자기 하루 비었네 ㅋㅋ", "reply_zh_tw": "直接當賺到一天假。", "reply_ko": "그냥 하루 휴가 생겼다고 치자.", "send_image": False}
        ],
        "avoid_when": ["실제 해고·실직 통보 직후", "일자리 불안을 진지하게 상담할 때"],
        "source_urls": ["https://www.instagram.com/readwhat.memes/p/Dfi_3R4Damn/"],
        "credit": "수집 게시자·이미지 워터마크: readwhat.memes / 已讀亂回迷因. 최초 농담 기원 미확인.",
        "evidence_status": "대만 계정 게시 이미지 확인. 기존 농담 구조이며 최신 바이럴로 확정하지 않음.",
        "asset_status": "게시 이미지 파일 그대로 다운로드. 채팅 형식의 밈이며 실제 대화 여부는 확인되지 않음.",
        "download_keys": ["thanks-boss"]
    },
    {
        "id": "holding-on-pirate", "title": "海盜船 — 일단 꽉 잡고 버티기", "meaning_ko": "편할 줄 알았는데 정신없이 휘둘리는 중",
        "format": "image", "tags": ["버티기", "정신없음", "예상과 현실", "여행", "월요일", "야근", "撐住", "快撐不住"],
        "tone": "고생하는 자기 모습을 과장해 보여주는 표정 반응",
        "image_files": ["images/pirate-remix.jpg", "images/pirate-original.jpg"],
        "image_text": "", "image_text_method": "visual inspection; 고정 자막 없음. 옷의 인쇄 문구는 반응 문구로 취급하지 않음.",
        "source_context_ko": "2026-11-05 林國漳의 강풍 속 유세 사진. 당사자가 해적선 같다고 표현했고 이용자들이 바이킹 놀이기구 등으로 합성했다.",
        "when_to_use": ["편한 일정이라고 생각했는데 빡센 상황이 됐을 때", "월요일·마감·여행 중 정신없이 버틴다는 가벼운 자조", "‘힐링이라며?’ 같은 예상과 현실 농담"],
        "image_fit": "기본 선택은 바이킹 합성본. 원 사진도 함께 보관해 출처·변형을 구분. 정치인 사진이라는 맥락은 남아 있음.",
        "how_to_use": "사진에 ‘說好的輕鬆行程呢？’ 같은 한 줄을 붙인다. 아래 일상 대화 용법은 수집자가 제안한 전용법으로, 실제 해당 문구의 유행을 주장하지 않는다.",
        "examples": [
            {"incoming_ko": "힐링 여행이라며? 새벽부터 산을 세 개 탔는데", "reply_zh_tw": "說好的輕鬆行程呢？", "reply_ko": "편한 일정이라며?", "send_image": True},
            {"incoming_ko": "월요일부터 회의가 여섯 개야 ㅋㅋ", "reply_zh_tw": "今天的我，先抓緊再說。", "reply_ko": "오늘의 나. 일단 꽉 잡고 보자.", "send_image": True}
        ],
        "avoid_when": ["진지한 사고·위기 상황", "인물 사진이 정치적 입장 표명으로 읽힐 대화"],
        "source_urls": ["https://www.threads.com/@lin.kuo.chang/post/DfWqBb6mquh/", "https://www.taisounds.com/news/content/71/298243"],
        "credit": "TaiSounds 게재 이미지, 기사 표기 출처 林國漳 Threads. 합성 제작자 별도 미확인.",
        "evidence_status": "원 사진과 실제 합성 변형 확인. 원문 임베드 좋아요 약 4.7만(2026-11-11 관측).",
        "asset_status": "기사에 게시된 원 사진 버전과 실제 유통 합성본을 각각 다운로드. 촬영 원본/RAW 파일을 뜻하지 않음.",
        "download_keys": ["campaign-1", "campaign-0"]
    },
    {
        "id": "tired-to-hyped", "title": "累 → 嗨", "meaning_ko": "피곤하다는 말을 신난다고 바꿔서 자조",
        "format": "text", "tags": ["피곤", "야근", "육아", "반어", "직장", "好累", "加班", "好嗨"],
        "tone": "억지로 긍정적인 척하는 피로 농담",
        "image_files": [], "image_text": None, "image_text_method": "not_applicable_text_meme",
        "source_context_ko": "말 속의 累를 嗨로 바꿔보자는 Threads 제안과 여러 변형 게시물. 고정 반응 이미지 없이 문장으로 재사용하는 형식.",
        "when_to_use": ["상대가 가볍게 피곤하다고 푸념하며 농담할 때", "내 야근·육아 피로를 반어적으로 말할 때"],
        "image_fit": "텍스트 전용. 원본 밈 사진을 확보한 항목이 아니며 가짜 원본 카드를 만들지 않음.",
        "how_to_use": "기존 문장의 累를 嗨로 바꾼다. 상대가 밈을 모르면 반어가 분명한 한 줄로 사용. 욕설 포함 변형은 친한 대화에서만 선택.",
        "examples": [
            {"incoming_ko": "오늘 야근 세 시간 했어 ㅋㅋ", "reply_zh_tw": "今天不是累，是嗨到快往生。", "reply_ko": "오늘은 피곤한 게 아니라 신나서 죽을 지경이네.", "send_image": False},
            {"incoming_ko": "오늘 일 어땠어?", "reply_zh_tw": "今天上班嗨到靠北。", "reply_ko": "오늘 일하면서 존나 신났지. [피곤했다는 반어]", "send_image": False}
        ],
        "avoid_when": ["상대가 위로를 원하며 심한 소진을 호소할 때", "상대가 진짜 즐겁다는 의미로 嗨를 쓰고 있을 때"],
        "source_urls": ["https://www.threads.com/@jakuziyong/post/Dfit6fMCY51/", "https://dailyview.tw/popular/detail/33934", "https://www.threads.com/@yeahitsme1048576/post/DfjB1R1Et7Z/"],
        "credit": "제안 게시자 黃少雍 Yong 및 참여 이용자들. 표현의 최초 발명 시점은 미확인.",
        "evidence_status": "현지 보도와 검색에서 변형 확인. 2026-11-11 11시 보도 기준 좋아요 7.3만·공유 2.8만. 원문 직접 전체 열람은 로그인 제한.",
        "asset_status": "텍스트 용법만 보관.", "download_keys": []
    }
]

downloads = json.loads((ROOT / 'evidence/downloads.json').read_text())
for card in cards:
    card['observed_on'] = '2026-11-11'
    card['usage_notes_basis'] = '원문 맥락을 바탕으로 수집자가 작성한 대화용 가이드. 예시 대화는 설명을 위한 작성 예시이며 실제 이용자 대화 인용이 아님.'
    card['assets'] = []
    for filename, key in zip(card['image_files'], card.pop('download_keys')):
        payload = (ROOT / filename).read_bytes()
        card['assets'].append({'file': filename, 'download_url_at_capture': downloads[key]['src'], 'sha256': hashlib.sha256(payload).hexdigest(), 'modified': False})

catalog = {
    'schema_version': 1,
    'locale': 'zh-TW',
    'created_on': '2026-11-11',
    'purpose_ko': '나중에 대화 스킬에서 상황에 맞춰 고를 수 있도록 보관한 반응 밈 자료집. 현재 자동 전송·스킬 설치·앱 연결은 하지 않음.',
    'selection_ko': '최신성만으로 고르지 않고 대화에서 재사용 가능한 상황을 우선. 대만에서 관측된 용법과 제안한 응용을 구분.',
    'future_selection_notes': [
        '단어 일치보다 대화 의도·감정·상대와의 말투를 먼저 판단한다.',
        '이미지 속 고정 문구가 현재 상황과 맞는지 확인한다. 표현만 맞으면 텍스트로 쓴다.',
        '실제로 해결해야 할 질문은 먼저 답한다. 밈은 필요할 때만 한 개 덧붙인다.',
        '사용자가 진지하게 힘들어하면 농담보다 공감과 실질적인 답을 우선한다.',
        '표현의 원래 용법과 새로 작성한 답장 예시를 혼동하지 않는다.',
        '파일을 사용할 때 출처·워터마크를 보존한다. 원본 자산의 권리·라이선스는 별도 확인되지 않음.'
    ],
    'cards': cards
}
(ROOT / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')

sections = []
for card in cards:
    photos = ''.join(f'<figure><a href="{E(f)}"><img src="{E(f)}" alt="{E(card["title"])}" loading="lazy"></a><figcaption>{E(Path(f).name)}</figcaption></figure>' for f in card['image_files'])
    if not photos:
        photos = '<div class="text-meme">累 → 嗨<br><small>텍스트용 · 고정 이미지 없음</small></div>'
    examples = ''.join(f'<div class="example"><p>상황: {E(x["incoming_ko"])}</p><strong lang="zh-Hant">{E(x["reply_zh_tw"])}</strong><p>{E(x["reply_ko"])}</p><small>{"이미지 함께 사용" if x["send_image"] else "텍스트만 사용"} · 작성 예시</small></div>' for x in card['examples'])
    links = ' · '.join(f'<a href="{E(u)}" target="_blank" rel="noreferrer">출처 {i+1}</a>' for i,u in enumerate(card['source_urls']))
    sections.append(f'''<article id="{card['id']}">
      <div class="visual">{photos}</div><div class="notes">
      <p class="eyebrow">{E(' / '.join(card['tags'][:4]))}</p><h2>{E(card['title'])}</h2><p class="meaning">{E(card['meaning_ko'])}</p>
      <p>{E(card['source_context_ko'])}</p><h3>언제 쓰나</h3><ul>{''.join('<li>'+E(t)+'</li>' for t in card['when_to_use'])}</ul>
      <h3>어떻게 쓰나</h3><p>{E(card['how_to_use'])}</p><p class="fit">{E(card['image_fit'])}</p>
      {examples}<p class="avoid"><b>안 맞는 상황</b> · {E(' / '.join(card['avoid_when']))}</p>
      <details><summary>원문 문구 · 출처 · 확인 범위</summary><pre>{E(card['image_text'] or '(고정 이미지 문구 없음)')}</pre><p>{E(card['asset_status'])}</p><p>{E(card['evidence_status'])}</p><p>{E(card['credit'])}</p><p>{links}</p></details>
      </div></article>''')

doc = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>대만 대화용 밈 라이브러리 v1</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f2ec;color:#212723;font:16px/1.7 system-ui,-apple-system,sans-serif}header,main,footer{max-width:1160px;margin:auto;padding:30px 26px}header{padding-top:55px}h1{font-size:34px;line-height:1.3;margin:8px 0}h2{font-size:25px;line-height:1.35;margin:6px 0}h3{font-size:16px;margin:22px 0 4px}p{margin:9px 0}a{color:#176349}nav{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}nav a{background:#e4e9df;border-radius:24px;padding:6px 14px;text-decoration:none;font-size:14px}article{background:white;border:1px solid #dedfd5;border-radius:18px;padding:24px;margin:0 0 28px;display:grid;grid-template-columns:minmax(0,0.9fr) minmax(0,1.2fr);gap:30px;scroll-margin-top:20px}.visual{min-width:0}figure{margin:0 0 16px}img{width:100%;height:auto;border-radius:10px}figcaption,small{color:#6b746a;font-size:12px}.eyebrow{color:#51765f;font-size:12px;letter-spacing:.05em}.meaning{font-size:18px;font-weight:600}.example{background:#f2f5ef;border-left:3px solid #759d79;padding:12px 16px;margin:12px 0}.example p{margin:3px 0}.example strong{font-size:19px}.fit{color:#4c6254;font-size:14px}.avoid{font-size:13px;color:#736955}details{border-top:1px solid #e2e6dc;padding-top:14px;margin-top:18px;font-size:13px}summary{cursor:pointer}pre{white-space:pre-wrap;font:inherit;background:#f7f7f2;padding:14px}.text-meme{padding:70px 20px;text-align:center;background:#eef2da;border-radius:10px;font-size:56px;font-weight:800}.text-meme small{font-weight:400}ul{padding-left:21px;margin:6px 0}footer{font-size:13px;color:#657064}@media(max-width:760px){article{grid-template-columns:1fr;padding:18px;gap:16px}header,main,footer{padding-left:16px;padding-right:16px}h1{font-size:28px}.visual{max-width:450px;margin:auto;width:100%}}</style>
<header><p class="eyebrow">TAIWAN REACTION LIBRARY · 2026-11-11 · v1</p><h1>대화하다 꺼내 쓰는 밈</h1><p>이미지 밈 4종 + 텍스트 밈 1종. 뜻·상황·답장 예시를 사진과 함께 기록했습니다.</p><p>이미지는 실제 게시 파일 5장입니다. ‘원본’은 수집한 게시 버전을 의미하며, 밈의 최초 창작 원본을 확정한 것은 아닙니다.</p><p><a href="catalog.json">스킬용 구조화 기록 JSON</a> · 예시 대화와 일상 전용법은 수집자가 작성했습니다.</p><nav>'''
doc += ''.join(f'<a href="#{c["id"]}">{E(c["title"])}</a>' for c in cards)
doc += '</nav></header><main>' + ''.join(sections) + '</main><footer>용도: 향후 대화 스킬을 위한 자료 보관. 현재 자동 사용은 연결하지 않았습니다. 최신 유행 여부와 대화에서의 유용성은 별도로 기록했습니다.</footer></html>'
(ROOT / 'index.html').write_text(doc)
(ROOT / 'README.txt').write_text('대만 대화용 밈 라이브러리 v1\n\nindex.html: 이미지와 용법을 함께 보는 카드 모음\ncatalog.json: 향후 스킬에서 검색할 구조화 기록\nimages/: 수정하지 않은 게시 이미지 파일 5장\nevidence/downloads.json: 수집 당시 이미지 URL\nbuild_catalog.py: 기록과 HTML 재생성 스크립트\n\n실제 관측 용법과 작성한 대화 예시는 구분되어 있습니다.\n이미지 속 고정 문구가 현재 대화와 맞는지 우선 확인합니다.\n앱 자동 전송이나 스킬 설치는 활성화하지 않았습니다.\n')
print(f'Built {len(cards)} cards with {sum(len(c["assets"]) for c in cards)} original published image files.')
