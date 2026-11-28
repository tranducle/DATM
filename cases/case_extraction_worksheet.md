# Retrospective Case Extraction Worksheet

This worksheet records the structured five-field manual extraction used for the three retrospective enterprise GenAI incidents analyzed in Section 5.2 and Section 5.3 of the manuscript. The machine-readable counterpart is `cases/case_extraction_worksheet.json`.

## Evidence-Tier Summary and Provenance Scope

All entries are strictly bounded to the four public sources cited in the manuscript. Only **1 of 3 cases** (Air Canada) rests on a primary legal document; **2 of 3 cases** (Chevrolet and Samsung) rely exclusively on secondary business or technology press reporting. No primary dealership injection transcript, primary Samsung internal memo, or private organizational log is cited or held.

| Case | Cited Sources | Evidence Tier | Primary / Secondary Count | Provenance Limit |
| :- | :- | :- | :- | :- |
| Air Canada | `moffatt2024aircanada` (BCCRT decision, 2024 BCCRT 149) | Primary legal ruling | 1 primary, 0 secondary | Grounded in one public tribunal ruling; no corroborating press reports cited and no proprietary chatbot logs available. |
| Chevrolet | `coldewey2023chevy` (Notopoulos, *Business Insider*, Dec 2023) | Secondary news report | 0 primary, 1 secondary | Single secondary business-press report; no primary dealership transcript or vendor log cited or held. |
| Samsung | `dobberstein2023samsung`, `karabus2023samsungban` (*The Register*, Apr and May 2023) | Secondary trade-press reports | 0 primary, 2 secondary | Two secondary trade-press articles from the same outlet; no primary internal memo cited or held; details remain hedged as reported. |

## Case 1: Air Canada Passenger Chatbot (Autonomous Integrity Failure)

- **Evidence tier:** Primary legal ruling (`primary_legal_record`; 1 primary source, 0 secondary sources).
- **Cited public source:**
  - `moffatt2024aircanada`: British Columbia Civil Resolution Tribunal, *Moffatt v. Air Canada*, 2024 BCCRT 149 (Feb 2024), <https://www.canlii.org/en/bc/bccrt/doc/2024/2024bccrt149/2024bccrt149.html>.
- **Unavailable primary artifacts:** Proprietary chatbot runtime telemetry, prompt configuration, and internal model logs are not part of the public tribunal decision.
- **Provenance limit:** Grounded in one public tribunal decision (2024 BCCRT 149). No corroborating secondary press reports are cited in the manuscript, and no proprietary system logs or historical replay are available.
- **Extracted five fields:**
  1. **Actor and access path:** Customer interacting with Air Canada's public customer-service chatbot on the airline's website through the intended external support channel.
  2. **Prompt or payload characteristics:** Routine passenger inquiry about bereavement-fare eligibility; the chatbot generated inaccurate guidance describing a non-existent retroactive bereavement-fare policy contradicting Air Canada's published webpage policy.
  3. **Documented organizational consequences:** Customer financial loss after relying on the inaccurate chatbot guidance and an adverse British Columbia Civil Resolution Tribunal ruling holding Air Canada liable (*Moffatt v. Air Canada*, 2024 BCCRT 149).
  4. **Deterministic controls that remained formally intact:** Conventional website access controls and system integrity boundaries remained formally intact; the tribunal record describes inaccurate chatbot output rather than an external attacker modifying code or infrastructure.
  5. **DATM-relevant runtime behavior:** Divergence between policy-authorized bereavement guidance (`y_pol`) and observed chatbot output (`y_exec`), illustrating the drift-frequency dimension of Generative Dissonance and Continuous Intent Auditing as an analytic lens rather than proof of historical prevention.

## Case 2: Chevrolet Dealership Chatbot (Semantic Prompt Injection)

- **Evidence tier:** Secondary news report (`secondary_news_report`; 0 primary sources, 1 secondary source).
- **Cited public source:**
  - `coldewey2023chevy`: Katie Notopoulos, *A Car Dealership Added an AI Chatbot to Its Site. Then All Hell Broke Loose*, Business Insider (Dec 2023), <https://www.businessinsider.com/car-dealership-chevrolet-chatbot-chatgpt-pranks-chevy-2023-12>.
- **Unavailable primary artifacts:** Primary dealership chatbot server transcript; chatbot vendor prompt configuration and API logs; official dealership or vendor incident report.
- **Provenance limit:** Supported solely by one secondary news article (Notopoulos, *Business Insider*, December 2023). No primary dealership injection transcript, vendor log, or independent corroborating source is cited or held; all incident details remain bounded to public press reporting.
- **Extracted five fields:**
  1. **Actor and access path:** As reported in public business-press coverage, an external user interacting with a Chevrolet dealership's public customer-service chatbot through its intended web chat interface.
  2. **Prompt or payload characteristics:** Reported natural-language instructions telling the chatbot to agree with the user's statements and append a "legally binding offer" phrase, followed by a reported $1 offer for a 2024 Chevrolet Tahoe.
  3. **Documented organizational consequences:** Reported generation of an unintended "$1 legally binding offer" response in a public customer-facing channel, creating reputational exposure and demonstrating unguarded conversational drift.
  4. **Deterministic controls that remained formally intact:** The reported manipulation used ordinary natural-language instructions through the public chatbot's intended interface rather than a conventional software exploit or perimeter compromise.
  5. **DATM-relevant runtime behavior:** Natural-language instructions shifting a domain-scoped dealership assistant away from benign automotive inquiries, illustrating Probabilistic Boundary Mapping (Equation 1) and motivating the constructed Phase A analog (not a historical replay).

## Case 3: Samsung ChatGPT Integration (Ingestion-Based Data Exfiltration)

- **Evidence tier:** Secondary trade-press reports (`secondary_trade_press`; 0 primary sources, 2 secondary sources).
- **Cited public sources:**
  - `dobberstein2023samsung`: Laura Dobberstein, *Samsung Reportedly Leaked Its Own Secrets through ChatGPT*, The Register (Apr 2023), <https://www.theregister.com/software/2023/04/06/samsung-reportedly-leaked-its-own-secrets-through-chatgpt/895536>.
  - `karabus2023samsungban`: Jude Karabus, *Samsung Puts ChatGPT Back in the Box after 'Code Leak'*, The Register (May 2023), <https://www.theregister.com/software/2023/05/02/samsung-to-ban-staff-from-using-chatgpt-after-code-leak/655856>.
- **Unavailable primary artifacts:** Primary Samsung internal corporate memo; internal data-loss-prevention or network telemetry logs; official primary corporate incident disclosure.
- **Provenance limit:** Supported solely by two secondary trade-press reports from the same publication (*The Register*, April and May 2023). No primary Samsung internal memo or corporate disclosure is cited or held; all incident details remain hedged as reported.
- **Extracted five fields:**
  1. **Actor and access path:** As reported in trade-press coverage, authorized Samsung employees using an external generative-AI service (ChatGPT) during work tasks.
  2. **Prompt or payload characteristics:** Employees reportedly entered sensitive source code, equipment-related information, and meeting records into prompts submitted to ChatGPT.
  3. **Documented organizational consequences:** Reported external disclosure of sensitive internal material, followed by a reported temporary restriction on staff use of external generative-AI tools.
  4. **Deterministic controls that remained formally intact:** The reported leakage occurred because authorized employees themselves submitted sensitive content through an external AI service rather than because an attacker breached a classic network perimeter.
  5. **DATM-relevant runtime behavior:** Authorized-channel semantic transmission of protected code-like or internal material without a perimeter intrusion, illustrating combined boundary and intent auditing in DATM and motivating the constructed Phase C synthetic analog.
