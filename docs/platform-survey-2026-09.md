# 플랫폼 조사 2026-09 — 오픈사이언스·문헌탐색 축

`similar-projects-survey.md`(2026-09-03)는 **딥리서치 에이전트**만 봤다: 프로세스를
어떻게 보여주는가가 관심사였고, 결론은 파생 트리였다. 이 문서는 다른 축을 본다 —
**연구 인프라, 문헌 탐색·스크리닝, 읽기 인터페이스, 인용 의미론, 증거·출처 표준.**
"Open Research / Open Science"라는 말이 실제로 가리키는 쪽이다.

조사 시점 2026-09-09. 수치는 각 출처가 스스로 밝힌 값.

---

## 1. 본 것들

### 1-1. 에이전트형 과학 플랫폼

| 플랫폼 | 핵심 | 우리와의 관계 |
|---|---|---|
| **Ai2 Asta** ([blog](https://allenai.org/blog/asta)) | Paper Finder(질의 분해→인용 추적→관련성 판정) + Scholar QA + **AstaBench** + 2억 편 정규화 코퍼스 API. "every output is cited and traceable to its sources", 평가는 **정확도-비용 파레토**와 **date-restricted retrieval**(재현성) | 철학이 가장 가깝다. 다만 인용은 *출처 링크*지 *검증된 문장*이 아니다 |
| **OpenScholar** | 사후 인용 귀속(retrieval-augmented 합성) | 같은 계열, 코퍼스 미보유 |
| **PaperQA2** | Search → **Gather Evidence** → Answer 3단 분해 | 근거 원장을 1급 단계로 (이전 조사에서 이미 지목) |
| **Elicit** | **증거 테이블**: 행=논문, 열=사용자가 정의한 추출 필드(최대 40열, 최대 4만 편). 자체 보고 recall 95%, abstract screening 감도 96.9% | 우리 *Problem 3*(선택한 세트 보여주기)의 가장 직접적인 선례 |
| **Undermind** | 반복 검색 라운드를 **소진할 때까지** 돌리고 recall을 주장 | "얼마나 훑었는지"를 제품이 말한다는 발상 |
| **Consensus** | claim 단위 답 + consensus meter, 저널 등급 필터 | 우리 가드레일과 정면 충돌(아래 §4) |

### 1-2. 스크리닝 — "6,600편에서 40편"의 정공법

**ASReview LAB v2** ([논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12416088/), Nature MI 2021 검증):
능동학습 스크리닝의 오픈소스 표준. 배울 점이 구조에 있다.

- **AI는 랭커, 인간이 oracle** — "AI serves as a ranker, never an autonomous filter"
- **정지 규칙과 진행 표시**: "연속 100건 무관하면 중단/전환", 화면의 *stopping circle*
- **에이전트 핸드오프**: 초기엔 TF-IDF+SVM, 100라벨 이후 의미 모델로 승계
- **감사 로그**: 레코드 id·라벨·타임스탬프·사용자·태그, 학습 사이클 메타데이터를
  남기되 모델은 저장하지 않는다 — "재학습해서 검증 가능한 만큼만" 기록
- 시뮬레이션 모드: 라벨된 데이터로 WSS@95 등 지표 재현

### 1-3. 읽기 인터페이스 — 우리 3색 카드의 선례

**Semantic Reader / Scim** ([TiiS 2024](https://dl.acm.org/doi/full/10.1145/3665648), [arXiv](https://arxiv.org/pdf/2205.04561)):
논문 본문에 **수사적 역할별 색 하이라이트**(objective / novelty / method / result)를
자동 표시. 연구 프로토타입에서 **521,000편 프로덕션**으로 확장, 31명 2회 사용자 연구.
설계 원칙 세 가지가 우리에게 직접 적용된다:

1. **facet = 색** (우리 pink/yellow/blue와 같은 계열)
2. **밀도를 독자가 조절**한다 (configurable density)
3. **논문 전체에 고르게 분포**시킨다 (evenly-distributed across a paper)

**CiteSee**: 인용에 "내가 전에 본 논문인가"라는 개인 이력 맥락을 덧입힘.

### 1-4. 인용 의미론

**scite** ([QSS 2021](https://direct.mit.edu/qss/article/2/3/882/102990/)): 2,500만 편 전문에서
**8.8억 개 인용 문장**을 뽑아 supporting / contrasting / mentioning으로 분류. 핵심 가치의
절반은 분류가 아니라 **인용 문맥 문장을 그대로 보여주는 것**에 있다.

**Inciteful**: 무료·무제한 인용 네트워크. **Literature Connector** — 두 논문 사이 최단
인용 경로. **Connected Papers**(공유 참고문헌 기반 유사도), **ResearchRabbit**(다중 시드),
**Litmaps**(x=연도, y=피인용) 은 시드→이웃 탐색의 세 변주.

### 1-5. 학회·피드

- **MiniConf** ([arXiv](https://arxiv.org/pdf/2007.12238)): 가상 학회 프레임워크. 논문을
  **임베딩 투영 점**으로 배치하고 **박스 선택 + facet 하이라이트**로 부분집합 분석.
  ICML 가상 사이트의 조상이며, 우리가 파싱하는 피드의 출신.
- **Paper Copilot** ([arXiv](https://arxiv.org/html/2510.13201v1), 20만+ 사용자): OpenReview
  API + 스크래핑 + 저자 제보로 **피어리뷰 동태**(점수 분포·연도별 통계)를 축적. 학회
  데이터에서 우리와 가장 인접한 실사용 서비스지만, 보는 축이 다르다(리뷰 프로세스 vs 내용).
- **Scholar Inbox** ([arXiv 2504.08385](https://arxiv.org/abs/2504.08385)): 개인화 피드.
  **콜드스타트를 능동학습으로** 푼다(몇 편 평가시켜 선호 학습) + "map of science" 개관,
  80만 개 사용자 평점 데이터 공개.
- **alphaXiv**(줄 단위 토론), **Emergent Mind**(트렌드 요약), **S2 Research Feeds**.

### 1-6. 인프라·표준

- **OpenAlex** 2.09억 works / **Semantic Scholar** 2.14억 편 — 우리 6개 에디션 3만 편은
  그 안의 티끌이지만, **전문·추출·검증까지 보유한** 3만 편이다.
- **nanopublication**(주장+출처를 인용 가능한 최소 단위로), **RO-Crate**(연구 객체 번들),
  2026년 "agent-native research artifacts" 논의 — AI가 만든 산출물의 **결정 이력과
  검증 사다리**를 기계가 읽을 수 있게 남기자는 흐름.
- **MCP 생태계**: arXiv/S2/OpenAlex/PubMed용 MCP 서버가 이미 다수(paper-search-mcp,
  Scientific-Papers-MCP, semantic-scholar-mcp …). 전부 **원격 검색 게이트웨이**다.
  로컬 보유 코퍼스 + 검증기를 도구로 내놓는 서버는 조사 범위에서 우리뿐.

### 1-7. 평가 문헌 — 우리 불변식의 외부 근거

2026년 딥리서치 에이전트 인용 품질 연구들이 일관되게 말한다:

| 출처 | 수치 |
|---|---|
| [Detecting and Correcting Reference Hallucinations](https://arxiv.org/html/2604.03173v1) | 상용 모델·에이전트의 인용 환각 **11~57%** |
| DRBench / FACT | 인용 정확도 78%(OpenAI DR) ~ 94%(Claude+search) |
| DRACO (프로덕션 질의 100건) | 최고 시스템도 **인용 품질 65%**, 사실성 68% |
| [Who is the Agent to Blame?](https://arxiv.org/abs/2608.24306) | 최종 보고서 오류의 84.7%가 오케스트레이터에서 발생 |

우리 종합 점검 실측은 앵커 62/63(98.4%)이고, 실패 1건은 **검증기가 에이전트의 미세
편집을 정상 차단**한 것이었다. 같은 축의 수치가 아니다 — 저들은 사후 판정, 우리는
**표시 전 차단**이다. 이 차이를 제품이 말하지 않고 있다(§3-A).

---

## 2. 우리 자리 — 무엇이 이미 있고 무엇이 없나

| | 우리 | 조사한 것들 |
|---|---|---|
| 코퍼스 | 6개 에디션 **전문 보유**, 추출·검증 완료 | 2억 편 인덱스(메타데이터 중심), 전문은 부분 |
| 인용의 지위 | **표시 전 서버 검증된 verbatim 문장** | 링크 또는 사후 귀속 |
| 재현성 | 코퍼스 고정 → 같은 질문 같은 근거 | Asta만 date-restricted retrieval로 흉내 |
| 실행 계정 | 사용자 본인 CLI, API 키 없음 | 전부 서비스형 |
| **세트 뷰** | 없음 (Problem 3 미완) | Elicit 증거 테이블, MiniConf 산점도 |
| **스크리닝 루프** | 없음 | ASReview 표준 워크플로 |
| **인용 문맥** | 엣지만(citations.json), 문장 없음 | scite 8.8억 문장 |
| **개인화/피드** | 없음 (세션형) | Scholar Inbox, S2 Feeds |
| 상호운용 | 없음 (BibTeX·CSV·Zotero 내보내기 전무) | 대부분 기본 제공 |

---

## 3. 개선 항목 (우선순위)

### A. 검증을 지표로 만들기 — "표시 전 차단"을 말하지 않고 보여주기 [작음]

근거: §1-7 수치, Asta "every claim … clickable citation".

1. **앵커 없는 주장 문장 표시**. 지금은 프롬프트로만 요구한다. `segment()`에서
   문장 단위로 쪼개, 논문을 지칭하면서 앵커가 없는 문장에 옅은 표시(설명 문구 금지,
   ✓/✗와 같은 계열의 마크). 가드레일 2 유지 — 판단이 아니라 앵커 유무라는 사실.
2. **검증 회귀 스위트**: `scripts/audit/`에 고정 질문 N개 + 기대 검증률 하한.
   프롬프트나 검증기를 건드릴 때마다 수치가 움직이는지 본다.
3. 대화 문서에 **재현 정보**(도구 호출 인자, 번들 버전, 코퍼스 스냅샷)를 저장하고
   답변 하단에 번들 id 한 줄. 같은 질문 재실행 버튼.

### B. 세트 뷰 = 증거 테이블 + 스크리닝 루프 [큼, 가장 밀린 것]

근거: Elicit 증거 테이블, ASReview 스크리닝, MiniConf 박스 선택.

Problem 3은 CLAUDE.md가 "least built"라고 적어둔 그대로 아직 없다. 세 조각으로:

1. **테이블 모드**: 행=선택한 논문, 열=추출 필드(proposes / builds on / data /
   tasks / limitation 용어 / 학회-연도). 전부 추출·계산값 → 가드레일 2·5 충족.
   정렬은 열 기준(사전순·개수)만, 점수 없음.
2. **읽기 큐**: 각 행에 "읽음 / 보류 / 제외" 표시. **ASReview의 규율을 그대로** —
   AI는 순서를 제안하지 않고, 표시한 것만 남는다. 진행 표시(40편 중 12편)와
   "제외 사유" 태그(자유 텍스트 아닌 칩). 저장은 대화 문서 옆에.
3. **선택 세트 산점도**(MiniConf식, 전체 코퍼스 금지 — CLAUDE.md 비목표): 이미
   `landscape.json` 좌표와 선택-스케일 재조정 규칙이 있다. 박스 선택 → 하위 세트.

### C. 인용 문맥 문장 [중간]

근거: scite. 우리는 이미 `fulltext_*.jsonl`과 참고문헌 파싱을 갖고 있다.

- A가 B를 인용한 **그 문장을 verbatim으로** 추출해 `citations.json` 엣지에 붙인다.
  분류(supporting/contrasting)는 **하지 않는다** — 모델 판단이므로 가드레일 1 위반.
- compare 뷰의 "the paper it cites" 한 줄 아래에 그 문장. 세트 사이드바의 STANDS ON
  행에도 "이 논문들이 그 조상을 어떻게 부르는지" 문장 2~3개.
- 검증기 소스에 인용 문맥도 포함시키면 에이전트가 인용 문장까지 근거로 쓸 수 있다.

### D. 상호운용 — 나가는 길 [작음, 효용 큼]

근거: Zotero 생태계, RO-Crate/nanopub.

1. 인용 칩 → **BibTeX 복사**, 선택 세트 → `.bib` / `.csv` 내보내기(Zotero·Excel).
2. 발행 대화 → **RO-Crate 유사 번들**(질문, 도구 호출, 인용+검증 결과, 코퍼스 버전).
   "agent-native research artifact" 흐름과 정합하고, 우리는 이미 그 재료를 다 갖고 있다.
3. `union.json`에 **OpenAlex / S2 id 매핑** 추가 → 외부 도구와 조인 가능.
   코퍼스 밖은 밖이라고 정직하게 말하는 것이 조인의 전제.

### E. MCP 서버를 "검증 서비스"로도 열기 [작음]

근거: MCP 문헌 서버들은 전부 원격 검색 게이트웨이. 우리만 로컬 보유 + 검증기.

- `verify_quote(gid, quote)` 도구 공개 — 다른 에이전트(Claude Code, Codex, orx)가
  자기 답변을 우리 코퍼스에 대고 검사할 수 있다. 우리 제품의 핵심을 도구화하는 것.
- `field_cards`/`gap_scan` 응답에 이미 있는 커버리지 문구를 **UI에도** 노출(§F).

### F. 커버리지·수치의 단일 출처 [작음, 지금 틀린 곳 있음]

`union.json`은 **29,605**(gid를 가진 논문), 사이트 빌드는 **29,669편**을 센다. 차이 64편은
초록이 없어 임베딩·gid가 없는 논문이다. 지금 README·Settings·MCP 설명·시스템 프롬프트에
29,605가 하드코딩돼 있고, browse는 29,669를 보여준다. 둘 다 맞지만 **어느 쪽인지 말하지
않는다** — 가드레일 4 위반에 가깝다.

- 빌드가 쓰는 `coverage` 산출물 한 곳에서 두 수를 함께 내보내고, 모든 표면이 그것을 읽게.
- 문장은 "29,669편 중 29,605편이 유사도·인용 검색 대상" 형태.

### G. 개인화는 "저장된 필드"로 최소하게 [중간, 뒤로]

근거: Scholar Inbox 콜드스타트 능동학습, S2 Research Feeds.

- 완전한 추천 시스템은 우리 성격이 아니다(랭킹 금지). 대신 **칩 조합 = 내 분야**를
  이름 붙여 저장하고, 새 에디션이 들어오면 **그 세트의 델타**만 보여준다.
- 시드 논문 기반 "field by example"은 이미 임베딩으로 가능 — 저장이 없을 뿐이다.

### H. 읽기 지원 — Scim의 두 가지만 [작음]

- **하이라이트 밀도/역할 토글**: 카드에서 pink/yellow/blue 중 무엇을 볼지. 세트를
  훑을 때 "한계만" 또는 "결과만" 보는 것이 실제 읽기 방식이다.
- **섹션 고른 분포**: `paper_text`가 섹션을 읽을 수 있으므로, 심층 카드에 섹션당 한
  문장(검증된 것만). Scim이 측정으로 확인한 "고르게 분포" 원칙.

---

## 4. 채택하지 않을 것 (그리고 왜)

| 남들이 하는 것 | 왜 안 하나 |
|---|---|
| Consensus 미터 / 주장 방향 집계 | 모델이 논문의 입장을 판정 → 가드레일 1·2 |
| scite의 supporting/contrasting **분류** | 같은 이유. 문맥 **문장**만 가져온다 |
| 저널 등급·피인용 필터(Consensus, Litmaps y축) | 중요도 랭킹 금지. 우리는 구조만 |
| Elicit·Consensus식 **모델 작성 요약** | 비목표 1번. 추출 문장이 우리 제품 |
| Scholar Inbox식 트렌딩/개인 점수 | 랭킹. 저장된 필드의 **델타**로 대체(§G) |
| 2억 편 원격 인덱스 통합 | 검증 불가능한 문장이 섞이는 순간 불변식이 죽는다. 밖은 링크로만 |
| OpenReview 리뷰 점수(Paper Copilot) | 다른 축(리뷰 프로세스)이고, 우리 CLAUDE.md의 dead end 기록 |

---

## 5. 한 줄 결론

조사한 어떤 플랫폼도 **보유 전문 + 표시 전 검증 + 커버리지 명시**를 동시에 갖고 있지
않다(2026-09 기준, 이전 조사와 동일한 결론). 반면 **세트를 보여주는 법**(Elicit 테이블,
ASReview 큐, MiniConf 선택)과 **밖으로 나가는 길**(BibTeX/Zotero/RO-Crate)은 저들이 오래
다듬었고 우리에겐 아직 없다. B와 D를 먼저, A·F를 그 다음에.
