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

## Breif Overview of How the Two Aystems Work

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

**Corpus size:** [TODO: Number of works and approximate word or token count, stating the unit.]

Full texts are also included in the GITHUB repository 

## Configuration

Record the settings used for the reported benchmark, including settings that differ between indexing and answering.

| Setting | VectorRAG | GraphRAG |
| --- | --- | --- |
| Framework and version | [TODO: LangChain and Chroma versions] | Microsoft GraphRAG v3.2.0 |
| Embedding model | text-embedding-3-large | text-embedding-3-large |
| Answer-generation model | gpt-4.1-mini | gpt-4.1-mini |
| Indexing / extraction model | Not applicable unless used | gpt-4.1-mini for graph extraction, description summarization, and community report generation |
| Chunk size and unit | [TODO: Value; characters or tokens] | 600 tokens, using the o200k_base encoding |
| Chunk overlap and unit | [TODO: Value and unit] | 100 tokens |
| Retrieval settings | [TODO: Search method and number of passages] | LanceDB vector store at output\lancedb. Local, global, DRIFT, and basic search are configured. The method used in the benchmark, community level, and query context limits are not specified in this file. |
| Answer instructions | [TODO: Prompt file and evidence rules] | Search-specific prompt files in prompts/, listed below. Their instructions and evidence rules require inspecting those files. |

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
| --- | --- |
| local_fact_retrieval | 20 | Retrieve specific definitions, concepts, or explanations from an individual work. |
| cross_document_reasoning | 20 | Connect and compare ideas across multiple works. |
| global_sensemaking | 20 | Synthesize broad themes and relationships across Rousseau’s writings. |


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

## Author

Killian K. Williams

**Portfolio Link**
https://killianportfolio6.wordpress.com/graphrag-write-up/
