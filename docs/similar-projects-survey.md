# 유사 오픈소스 프로젝트 조사 (2026-09-03)

리서치 에이전트 계열 고스타 프로젝트에서 차용할 기능/컨셉 조사.
스타 수는 조사 시점 GitHub 페이지 실측.

| 프로젝트 | ★ | 성격 | 프로세스 표시 | 차용 후보 |
|---|---|---|---|---|
| DeerFlow (bytedance) | 81.3k | 딥리서치→슈퍼에이전트 | **실행 전 계획 카드 + 자연어 수정/승인**, 활동 패널, 세션 리플레이 | 계획 승인 게이트, 활동/결과물 좌우 분리 |
| GPT Academic | 71.3k | 논문 읽기/번역 UI | 청크 진행률 정도 | 선언적 플러그인 버튼 |
| Khoj | 37k | 셀프호스팅 두 번째 뇌 | train-of-thought 접이식 | 로컬 코퍼스+챗 제품형 |
| Onyx | 31.9k | 사내지식 챗 | 다단계 리서치 순차 표시 | 질문→다단계→인용 보고서 뼈대 |
| STORM/Co-STORM | 31.2k | 인용 장문 보고서 | **아웃라인-먼저**, Co-STORM **갱신형 마인드맵** | 마인드맵=limitation 클러스터 트리의 선례, 관점 분해 |
| GPT Researcher | 29.3k | 자율 리서치 원조 | **질문 분해 목록 선출력** + 단계 스트리밍, 재귀 트리 | 서브질의 노출 문법 |
| Kotaemon | 25.7k | 문서 RAG UI | **인용→원문 하이라이트 왕복**, 저관련 경고 | verify_spans의 UI 대응물 |
| deep-research (dzhng) | 19.6k | 미니멀 재귀 CLI | breadth×depth 트리 파라미터화 | learnings 누적 루프 |
| SurfSense | 16.1k | NotebookLM 대체 | 산출물 중심 | 결과의 지식베이스 적립 |
| AI-Scientist | 14.5k | 자율 과학연구 | 단계별 아티팩트 | 스테이지별 검사가능 파일 규율 |
| open_deep_research | 12.7k | LangGraph 딥리서치 | 그래프 뷰, 섹션 계획 승인(legacy) | 팬아웃 그래프 |
| PaperQA2 | 9.1k | 과학문헌 고정밀 RAG | Search→**Gather Evidence**→Answer 분해 | 근거 원장을 1급 단계로 |

참고: openresearch-cli 688★(git 실험 트리·계보), OpenScholar 1.6k★(사후 인용 귀속 — 철학 최근접).

## 지배적 프로세스 시각화 패턴 4가지

1. **계획-우선 + 승인 게이트** (DeerFlow, GPT Researcher, STORM)
2. **스트리밍 활동 피드** (GPT Researcher, Khoj, Onyx) — 우리의 도구 궤적이 이미 이 패턴
3. **탐색 트리/마인드맵** (Co-STORM, deep-research, open_deep_research, orx 실험 트리)
4. **근거 원장** — 답 이전에 근거 집합 확정, 인용→원문 왕복 (PaperQA2, Kotaemon)

## 블루오션 파생 플로의 설계 결론

플로(카드 다독 → limitation 클러스터 → 응답 key_change 부재 확인 → gap 후보)는
집합 연산의 깔때기라 채팅 로그(패턴 2)로는 부족 — **패턴 3+4 결합**이 정답:

- 실행 전: 패턴 1로 "어느 필드 몇 장을 훑겠다" 계획 카드 1장 승인
- 실행 중/후: **파생 트리** — 루트=선택 필드, 가지=limitation 클러스터(대표
  문장은 verbatim 핑크 스팬), 잎=그 클러스터에 key_change로 응답하는 카드.
  **응답 잎이 빈 가지가 곧 gap 후보.** 노드마다 읽은 장수·클러스터 수·coverage
  명시(가드레일 4), 정렬은 점수가 아니라 클러스터 크기·응답 부재라는 측정
  구조로만(가드레일 1)
- 문장/잎 클릭 → 해당 카드 하이라이트로 착지 (Kotaemon식, 이미 /browse#p 보유)

조사한 12개 중 "검증 verbatim 인용 + 완전 보유 코퍼스 + coverage 명시"를 모두
갖춘 것은 없음 — 그 결합이 우리의 자리.
