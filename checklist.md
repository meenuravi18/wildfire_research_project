## CIKM-Readiness Checklist
The most interesting part of your paper will likely be the ablation study. CIKM reviewers will want to see the Pareto frontier between Precision (Correct Jurisdiction) and Recall (Information Coverage).

Does the "Full Geo-First" method sacrifice too much information to stay safe?

Visualizing this tradeoff 

Section,Focus
Introduction,"Define the ""Jurisdictional Hallucination"" problem in RAG."
Problem Formalization,Formally define the Geographic Knowledge Graph and the Hierarchy Distance metric.
Proposed Method,Detail the Geo-First ranking and the Warn/Abstain logic.
Benchmark,"Describe your wildfire dataset (the ""Wildfire-GeoBench"")."
Experiments,The 5-way method comparison + LLM-as-a-judge metrics.
Discussion,"The reliability-coverage tradeoff and ""Good-Decision Rate."""

### 1. Reliability Framework

- [ ] Name the framework, for example **GeoRisk-RAG** or **Geography-Aware Evidence Reliability**.
- [ ] Define the failure mode: semantically similar evidence can be geographically or jurisdictionally wrong.
- [ ] Define question hierarchy: city → county → state → region → country.
- [ ] Define passage hierarchy from retrieved geo mappings.
- [ ] Define hierarchy distance.
- [ ] Define granularity match.
- [ ] Define evidence relation labels: same-scope, parent-scope, sibling/near-miss, unrelated.
- [ ] Define decision behavior: answer / warn / abstain.
- [ ] Explain that the full method uses geography first, then semantic ranking.

### 2. Benchmark and Ablation Study

- [ ] Report number of questions.
- [ ] Report location-dependent vs non-location split.
- [ ] Report jurisdiction levels: city, county, state, region, country, other.
- [ ] Report question types: warnings, codes, mitigation, smoke, recovery, etc.
- [ ] Describe sources: PDFs, news, local government pages, wildfire reports.
- [ ] Describe how hierarchies were created or validated.
- [ ] Include example benchmark rows.
- [ ] Evaluate all five methods:
  - [ ] Similarity-only
  - [ ] Exact keyword match
  - [ ] Granularity-only
  - [ ] Graph match
  - [ ] Full geo-first method

### 3. Evaluation Metrics

- [ ] Report automatic metrics:
  - [ ] mean answer similarity
  - [ ] median answer similarity
  - [ ] same-scope rate
  - [ ] parent-scope rate
  - [ ] sibling/near-miss rate
  - [ ] unrelated rate
  - [ ] mean geo distance
  - [ ] abstention rate
- [ ] Split results by:
  - [ ] location-dependent questions
  - [ ] non-location questions
  - [ ] method
  - [ ] model configuration
- [ ] Add LLM-judge metrics:
  - [ ] supported answer rate
  - [ ] wrong-confident answer rate
  - [ ] appropriate warning rate
  - [ ] appropriate abstention rate
  - [ ] over-abstention rate
  - [ ] good-decision rate
  - [ ] bad-decision rate

### 4. Statistical Validation and Examples

- [ ] Add bootstrap confidence intervals for:
  - [ ] wrong-confident answer rate
  - [ ] good-decision rate
  - [ ] same-scope rate
  - [ ] mean answer similarity
- [ ] Include qualitative examples for:
  - [ ] similarity-only retrieving wrong but semantically similar jurisdiction
  - [ ] exact keyword match failing when exact local evidence is missing
  - [ ] granularity-only matching same level but wrong place
  - [ ] graph match being geographically safe but conservative
  - [ ] full geo-first warning or abstaining appropriately
- [ ] Include target hierarchy and retrieved passage hierarchy in examples.

### 5. Paper Positioning and Claims

- [ ] Position the paper as **RAG reliability / knowledge-aware retrieval**, not just metadata filtering.
- [ ] Emphasize that geographic hierarchy is a knowledge structure.
- [ ] Make the central claim about reducing wrong-confident answers.
- [ ] Present full geo-first as a controllable reliability–coverage tradeoff.
- [ ] Avoid claiming the method beats every baseline on every metric.
- [ ] State limitations:
  - [ ] full geo-first may abstain more
  - [ ] exact keyword can be safest when exact evidence exists
  - [ ] LLM judge may be imperfect
  - [ ] hierarchy extraction can introduce errors
  - [ ] current benchmark is wildfire-focused
  - [ ] method should be tested on other location-dependent domains
