from pypdf import PdfReader
import os
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdfs(dir_path) -> dict:
    """
    Get a dictionary of PDF files in the specified directory with their text content.
    
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
        
        pages = [page.extract_text().strip() for page in reader.pages[:10]] # TODO: Change to reader.pages to read all pages
        pdf_pages[pdf] = pages

    return pdf_pages


def text_splitter(pdf_pages_dict, chunk_size=1000, overlap=100) -> dict:
    """
    Split text from PDFs into chunks.
    
    params
    -----
    pdf_pages_dict: dict
        A dictionary where keys are PDF file names and values are lists of text from each page.
    chunk_size: int
        The maximum character size of each text chunk.
    overlap: int
        The number of characters to overlap between chunks.

    returns
    -----
    dict: A dictionary where keys are PDF file names and values are lists of text chunks.
    """
    pdf_chunks_dict = {}
    for pdf, pages in pdf_pages_dict.items():
        combined_text = ''
        print(f"Number of pages: {len(pages)}")
        for page in pages:
            combined_text += ' ' + page
       
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.create_documents([combined_text])
        chunk_content = [chunk.page_content for chunk in chunks]
        pdf_chunks_dict[pdf] = chunk_content

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


def embed_pdf_chunks(pdf_chunks_dict) -> dict:
    """
    Embed chunks of text from PDFs using the Ollama embedding model.
    
    params
    -----
    pdf_chunks_dict: dict
        A dictionary where keys are PDF file names and values are lists of text chunks.

    return 
    -----
    dict: A dictionary where keys are PDF file names and values are lists of embeddings for the text chunks.
    """
    pdf_embeddings_dict = {}
    for pdf, chunks in pdf_chunks_dict.items():
        embeddings = embed_batch(chunks)
        pdf_embeddings_dict[pdf] = embeddings

    return pdf_embeddings_dict
    

if __name__ == "__main__":
    # Read PDFs from the 'Data' directory
    pdf_pages = read_pdfs('Data')

    # Split text from the PDFs into chunks
    pdf_chunks = text_splitter(pdf_pages)

    # Embed the chunks of text from the PDFs
    pdf_embedded_chunks = embed_pdf_chunks(pdf_chunks)

    # Example
    pdf_name = 'An Introduction to Statistical Learning with Applications in Python.pdf'
    example_chunks = pdf_chunks[pdf_name]
    example_embeddings = pdf_embedded_chunks[pdf_name]



