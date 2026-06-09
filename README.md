# RAG
Retrieval-Augmented Generation (RAG) is a technique to augment large language models (LLMs) with new information from selected datasources (typically unstructure data). This allows LLMs to reference source documents and provide more accurate and contex-aware responses to user queries. 

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/download) installed and running
- Install requirements.txt
- Pull `embeddinggemma` model (TODO: Can I add this to the requirements.txt somehow?)
    ```
    ollama pull embeddinggemma
    ```

## Data Preparation


## Future Features
- Semantic chunking (on header, paragraph, complete thought, etc) rather than by character length.