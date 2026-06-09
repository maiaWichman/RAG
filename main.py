from pypdf import PdfReader
import os
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        
        pages = [page.extract_text().strip() for page in reader.pages[:10]]
        pdf_pages[pdf] = pages

    return pdf_pages


def text_splitter(pdf_pages_dict, chunk_size=1000, overlap=100) -> dict:
    pdf_chunks_dict = {}
    for pdf, pages in pdf_pages_dict.items():
        combined_text = ''
        for page in pages:
            combined_text += ' ' + page
       
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )
        pdf_chunks_dict[pdf] = splitter.create_documents([combined_text ])

    return pdf_chunks_dict



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
    pdf_pages = read_pdfs('Data')

    chunks = text_splitter(pdf_pages)
    print(len(chunks['An Introduction to Statistical Learning with Applications in Python.pdf']))

    for chunk in chunks['An Introduction to Statistical Learning with Applications in Python.pdf'][:5]:
        print(chunk)
        print(len(chunk.page_content))
        print('---')



