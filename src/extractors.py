import os
import pandas as pd
import pdfplumber
import camelot
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PIL import Image
import pytesseract
from io import BytesIO
import warnings

# Suppress Camelot image-based page warnings
warnings.filterwarnings('ignore', message='.*is image-based, camelot only works on text-based pages.*')

# Set Tesseract path for Windows (adjust if installed elsewhere)
if os.name == 'nt':  # Windows
    tesseract_path = r'C:\Users\charb\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

def extract_all_from_pdf_page(page, page_num, use_camelot_tables=None):
    """Extract text, tables, and images from a single PDF page"""
    extracted_data = {
        'text': '',
        'tables': [],
        'ocr_text': []
    }

    # Extract regular text
    text = page.extract_text(layout=True)
    if text:
        extracted_data['text'] = text

    # Extract tables from pdfplumber
    tables = page.extract_tables()
    for i, table in enumerate(tables):
        if table:
            df = pd.DataFrame(table[1:], columns=table[0])
            table_text = f"\n[TABLE {i+1} on Page {page_num}]\n" + df.to_string(index=False) + "\n[END TABLE]\n"
            extracted_data['tables'].append(table_text)

    # Extract text from images using OCR
    try:
        images = page.images
        page_bbox = (0, 0, page.width, page.height)

        for i, img in enumerate(images):
            try:
                # Ensure bbox is within page bounds
                bbox = (
                    max(img['x0'], 0),
                    max(img['top'], 0),
                    min(img['x1'], page.width),
                    min(img['bottom'], page.height)
                )

                # Skip if bbox is invalid
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue

                cropped_img = page.within_bbox(bbox).to_image(resolution=300)
                pil_img = cropped_img.original

                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    extracted_data['ocr_text'].append(f"\n[IMAGE {i+1} OCR TEXT]\n{text.strip()}\n[END IMAGE {i+1}]\n")
            except Exception as e:
                print(f"Error extracting text from image {i+1} on page {page_num}: {e}")
    except Exception as e:
        print(f"Error accessing images on page {page_num}: {e}")

    # Add Camelot tables if available
    if use_camelot_tables and page_num in use_camelot_tables:
        extracted_data['tables'].extend(use_camelot_tables[page_num])

    return extracted_data


def extract_text_from_pdf_advanced(pdf_path):
    """Extract text, tables, and images from PDF in a single pass"""
    documents = []
    filename = os.path.basename(pdf_path)

    # Try Camelot first for better table extraction
    camelot_tables = {}
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for i, table in enumerate(tables):
            df = table.df
            table_text = f"\n[TABLE {i+1}]\n" + df.to_string(index=False) + f"\n[END TABLE {i+1}]\n"
            camelot_tables.setdefault(table.page, []).append(table_text)
    except Exception as e:
        print(f"Camelot table extraction failed: {e}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract everything in one pass
                page_data = extract_all_from_pdf_page(page, page_num, camelot_tables)

                # Combine all extracted content
                combined_text = page_data['text'] or ""
                if page_data['ocr_text']:
                    combined_text += "\n\n" + "\n".join(page_data['ocr_text'])
                if page_data['tables']:
                    combined_text += "\n\n" + "\n".join(page_data['tables'])

                if combined_text.strip():
                    doc = Document(
                        page_content=combined_text,
                        metadata={
                            "source": pdf_path,
                            "filename": filename,
                            "page": page_num,
                            "file_type": "pdf",
                            "extraction_method": "pdfplumber_unified",
                            "has_ocr": bool(page_data['ocr_text']),
                            "has_tables": bool(page_data['tables'])
                        }
                    )
                    documents.append(doc)
    except Exception as e:
        print(f"Advanced PDF extraction failed for {filename}: {e}. Falling back to PyPDFLoader.")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                'filename': filename,
                'file_type': 'pdf',
                'extraction_method': 'pypdf_fallback'
            })

    return documents


def extract_text_from_excel(excel_path):
    """Extract text from Excel file and convert to Document objects"""
    documents = []
    filename = os.path.basename(excel_path)
    excel_file = pd.ExcelFile(excel_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        text_content = f"[EXCEL SHEET: {sheet_name}]\n\n"
        text_content += f"Columns: {', '.join(df.columns.tolist())}\n"
        text_content += f"Total Rows: {len(df)}\n\n"
        text_content += "[TABLE]\n" + df.to_string(index=True) + "\n[END TABLE]\n\n"
        text_content += "[ROW-BY-ROW DATA]\n"
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            text_content += f"Row {idx + 1}: {row_text}\n"
        text_content += "[END ROW-BY-ROW DATA]\n"

        doc = Document(
            page_content=text_content,
            metadata={
                "source": excel_path,
                "filename": filename,
                "sheet": sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "file_type": "excel"
            }
        )
        documents.append(doc)

    return documents


def extract_text_from_multiple_files(file_paths):
    """Extract text from multiple PDF and Excel files"""
    all_documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            all_documents.extend(extract_text_from_pdf_advanced(file_path))
        elif ext in [".xlsx", ".xls"]:
            all_documents.extend(extract_text_from_excel(file_path))
    return all_documents


def split_chunk_overlap(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents with special handling for tables"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    return splitter.split_documents(documents)
