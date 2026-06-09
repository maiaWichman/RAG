from pypdf import PdfReader
import os


def _read_pdfs(dir_path) -> dict:
    """
    Get a list of PDF files in the specified directory.
    
    params
    -----
    dir_path: str - The path to the directory to search for PDF files.

    returns
    -----
    dict - A dictionary where keys are PDF file names and values are lists of text from each page.
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



