from pypdf import PdfReader
import os
import ollama

def read_pdfs(dir_path) -> dict:
    """
    Get a list of PDF files in the specified directory.
    
    params
    -----
    dir_path: str 
        The path to the directory to search for PDF files.

    returns
    -----
    dict: A dictionary where keys are PDF file names and values are lists of text from each page.
    """
    paths = os.listdir(dir_path)
    pdf_names = [f for f in paths if f.endswith('.pdf')]

    pdf_pages = {}
    for pdf in pdf_names:
        print(pdf)
        reader = PdfReader(os.path.join(dir_path, pdf))
        print(f"Number of pages in {pdf}: {len(reader.pages)}")
        
        pages = [page.extract_text() for page in reader.pages]
        pdf_pages[pdf] = pages

    return pdf_pages


def embed(text: str) -> list:
    """
    Embed text using the Ollama embedding model.
    
    params
    -----
    text: str
        The text to be embedded.

    return 
    -----
    list: A list of the embedding for the input text.
    """
    embedding = ollama.embed(
        model='embeddinggemma',
        input=text
    )

    return embedding['embeddings']


def embed_batch(texts: list) -> list:
    """
    Embed a batch of texts using the Ollama embedding model.
    
    params
    -----
    texts: list
        A list of strings to be embedded.

    return 
    -----
    list: A list of embeddings for the input texts.
    """
    batch = ollama.embed(
        model='embeddinggemma',
        input=texts
    )

    return batch['embeddings']

if __name__ == "__main__":
    pass
