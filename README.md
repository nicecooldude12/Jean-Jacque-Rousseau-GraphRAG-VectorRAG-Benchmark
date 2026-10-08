## Overview
The purpose of this project is to explore Microsoft's GraphRAG and VectorRAG ability to answer
questions on Jean-Jacques Rousseau's political/philisophical corpus.

## Thesis
**How do these two systems compare when answering questions about themes and ideas that span multiple documents?**

Rousseau's writings was chosen because his corpus contains multiple works that has the same overall theming/concept such
as freedom, rights, and government.
This project examines whether or not graph-based retrieval improves upon answer generation and also the 
trade-offs compared to VectorRAG.

## Contribution

Microsoft's GraphRAG system is used to a select books by Rousseau. The data was chosen from an open source
site Project Gutenburg

- Collected and prepared the source texts for indexing.
- Configured GraphRAG's models, text chunking, and prompts.
- Implemented a VectorRAG baseline using Python, LangChain, and Chroma database.
- Developed a benchmark question set covering specific details and broader political/philosophical themes.
- Built scripts to generate answers, evaluate their quality, and analyze response times.
- Documented the strengths and limitations of each approach.

## Breif Overview of How the Two Systems Work

| Stage | VectorRAG | GraphRAG |
| --- | --- | --- |
| Preparation | Splits source texts into passages and embeds them. | Splits texts, extracts entities and relationships, and builds a graph and community reports. |
| Retrieval | Finds passages similar to the question using vector search. | Uses the configured search method to retrieve graph-related context and/or community reports. |
| Answer generation | Supplies retrieved passages to a language model. | Supplies the selected context to a language model. |

GraphRAG supports different search methods. **The method evaluated here is [TODO: global, local, DRIFT, or another method actually used].** Describe its actual retrieval behavior rather than treating all GraphRAG searches as equivalent.

## Source Texts
Both GraphRAG and VectorRAG used these documents listed below. These were verified to be free domain. No copyrighted material was used.


Below list of sources used:

A Discourse Upon the Origin and the Foundation of the Inequality Among Mankind. Translator is unknown.  https://www.gutenberg.org/ebooks/5427
The social contract & discourses. Translated by Cole, G. D. H. (George Douglas Howard).  https://www.gutenberg.org/ebooks/46333
Emile. Translated by Barbara Foxley. https://www.gutenberg.org/ebooks/5427

**Text preparation:**
All sources were used in a .txt format for these systems to embed. 
Before use, anything that was not written by Rousseau himself was removed which includes ie Introductions, Headers, Footers etc.

## Configuration

Record the settings used for the reported benchmark, including settings that differ between indexing and answering.

| Setting | VectorRAG | GraphRAG |
| --- | --- | --- |
| Framework and version | LangChain with langchain_openai, langchain_chroma, and langchain_text_splitters; Chroma vector database. Versions not recorded in the supplied files. | Microsoft GraphRAG v3.2.0 |
| Embedding model | text-embedding-3-large | text-embedding-3-large |
| Answer-generation model | gpt-4.1-mini with temperature=0 | gpt-4.1-mini |
| Indexing / extraction model | Not applicable unless used | gpt-4.1-mini for graph extraction, description summarization, and community report generation |
| Chunk size and unit | 500 characters, using RecursiveCharacterTextSplitter | 600 tokens, using the o200k_base encoding |
| Chunk overlap and unit | Configured overlap of 100 characters | 100 tokens |
| Retrieval settings | Chroma vector store at chroma_db, collection spinoza_collection. Uses similarity_search_with_score(k=5) to retrieve up to five passages. No score threshold or reranking is applied. | LanceDB vector store at output\lancedb. Local, global, DRIFT, and basic search are configured. The method used in the benchmark, community level, and query context limits are not specified in this file. |
| Answer instructions | PROMPT_TEMPLATE in rag_chain.py: base answers on retrieved excerpts, address the question directly, explain concepts clearly, connect ideas when supported, distinguish evidence from interpretation, and avoid unsupported claims or invented quotations and citations. Answer partially when evidence is incomplete; explicitly state insufficient information when no answer is supported. | Search-specific prompt files in prompts/, listed below. Their instructions and evidence rules require inspecting those files. |

See `settings.yaml` for further details.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `RAG application/` | VectorRAG implementation, including ingestion and retrieval code. |
| `benchmark/` | Benchmark questions, answer-generation scripts, evaluation scripts, and small results files. |
| `prompts/` | Prompts used by the GraphRAG workflow. |
| `settings.yaml` | GraphRAG configuration. |
| `requirements.txt` | Python dependencies and versions needed to reproduce the project. |
| `.gitignore` | Excludes credentials, environments, and generated artifacts. |

Generated databases, caches, logs, and full indexing outputs are excluded. They can be recreated locally.

## Evaluation Method

**Question set**

The benchmark contains 60 questions that are evenly divided into three categories:

| Category | Questions | Purpose |
| --- | --- | --- |
| local_fact_retrieval | 20 | Retrieve specific definitions, concepts, or explanations from an individual work. |
| cross_document_reasoning | 20 | Connect and compare ideas across multiple works. |
| global_sensemaking | 20 | Synthesize broad themes and relationships across Rousseau’s writings. |


## Results

The benchmark evaluated 60 questions, with 20 questions in each of three categories. Each answer pair was evaluated by gpt-4.1-mini on four criteria, with three judgments per criterion. The 720 completed judgments were aggregated into 240 question–criterion majority decisions, with no ties or incomplete comparisons.


| Measure | VectorRAG | GraphRAG |
| --- | --- | --- |
| Questions successfully answered / attempted | 60/60 | 60/60 |
| Comprehensiveness win rate | 0% (0/60) | 100% (60/60) |
| Diversity win rate | 0% (0/60) | 100% (60/60) |
| Empowerment win rate | 16.67% (10/60) | 83.33% (50/60) |
| Directness win rate | 75% (45/60) | 25% (15/60) |
| Overall win rate across four criteria | 22.92% (55/240) | 77.08% (185/240) |
| Mean response time, seconds | 5.419 | 656.376 |
| Median response time, seconds | 5.054 | 652.816 |
| Source-support / faithfulness score | Not measured | Not measured |
| Indexing time, minutes | Not measured | Not measured |
| Total API cost, USD | Not measured | Not measured |


## Results by Question Category
| Measure | VectorRAG | GraphRAG |
| --- | --- | --- |
| Local fact retrieval | 23.75% (19/80) | 76.25% (61/80) |
| Cross-document reasoning | 21.25% (17/80) | 78.75% (63/80) |
| Global sensemaking | 23.75% (19/80) | 76.25% (61/80) |

**Interpretation**
GraphRAG was preferred on comprehensiveness, diversity, and empowerment, while VectorRAG was preferred on directness. However, its mean response time was approximately 10.94 minutes, compared with 5.42 seconds for VectorRAG—approximately 121 times longer in this implementation.

These results indicate a significant trade-off between the judge’s preference for broader answers and the speed and directness of VectorRAG. Whether the additional waiting time is worthwhile depends on the intended application. Financial cost-effectiveness could not be assessed because total API costs were not calculated.

Response times reflect the implemented runners: GraphRAG timing includes command-line subprocess startup, while VectorRAG timing covers a direct function call. The judge was not supplied with source passages or verified reference answers, so preference rates do not establish factual accuracy, citation correctness, or faithfulness to Rousseau’s writings. Remaining differences in prompts, chunking, and context budgets also limit attribution of the results solely to retrieval architecture.

Supporting results: summary.json.

**Answer evaluation** 

Answers were evaluated using pairwise LLM-as-a-judge comparisons with gpt-4.1-mini. For each question, the judge compared the VectorRAG and GraphRAG answers on four criteria:
Suggested dimensions are relevance, coverage, and support from the source texts. 
A judge's preference alone does not establish factual accuracy.

| Criterion | Purpose |
| --- | --- |
| Comprehensiveness | How thoroughly the answer addresses the question. |
| Diversity | How effectively the answer helps the reader understand the topic and make informed judgments. |
| Empowerment | Synthesize broad themes and relationships across Rousseau’s writings. |
| Directness | How directly and clearly the answer addresses the question. |

The evaluation recorded a preferred answer and a written explanation rather than a numerical quality score. Each question received three judge comparisons per criterion, producing 720 completed comparisons: 60 questions × 4 criteria × 3 runs.

## Limitations

- Findings from this Rousseau corpus may not generalize to other authors or domains.
- Translation choices and text preparation can affect retrieval and interpretation.
- Extracted graph relationships and generated summaries can contain errors.
- Question set may favor particular question types.
- LLM-based evaluation can favor longer or more polished answers and may miss unsupported claims.
- Different corpora, answer models, or context budgets limit conclusions about retrieval alone.
- Response times vary with API conditions, caching, and models.

## Future Work
A list of suggestions to improve upon this project: 

- Expand the benchmark with more questions and source material such as academic papers on Rousseau's overall philosophy.
- Add human review of a sample of answers and citations.
- Compare additional GraphRAG search methods such as HypridRAG.
- Analyze API cost.
- Use up-to-date OpenAI models.

## References and Attribution

This project uses Microsoft's GraphRAG framework. The framework and underlying research belong to their respective authors; this repository documents my application and evaluation of the approach.


**Research Paper**
https://arxiv.org/abs/2404.16130

**Github Repository**
https://github.com/microsoft/graphrag

For VectorRAG framework, check the link below:

**Research Paper**
https://arxiv.org/abs/2005.11401

## Author

Killian K. Williams

**Portfolio Link**
This is
https://killianportfolio6.wordpress.com/graphrag-write-up/
