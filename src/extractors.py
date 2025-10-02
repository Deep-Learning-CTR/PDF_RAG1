import os
import pandas as pd
import pdfplumber
import camelot
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using Camelot and pdfplumber"""
    tables_data = []

    try:
        # Try Camelot first (better for complex tables)
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for i, table in enumerate(tables):
            df = table.df
            table_text = f"\n[TABLE {i+1}]\n" + df.to_string(index=False) + f"\n[END TABLE {i+1}]\n"
            tables_data.append({'page': table.page, 'text': table_text, 'type': 'camelot'})
    except Exception as e:
        print(f"Camelot table extraction failed: {e}. Trying pdfplumber...")

    # Fallback to pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for i, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_text = f"\n[TABLE on Page {page_num}]\n" + df.to_string(index=False) + "\n[END TABLE]\n"
                        tables_data.append({'page': page_num, 'text': table_text, 'type': 'pdfplumber'})
    except Exception as e:
        print(f"PDFPlumber table extraction failed: {e}")

    return tables_data


def extract_text_from_pdf_advanced(pdf_path):
    """Extract text and tables from PDF with enhanced structure preservation"""
    documents = []
    filename = os.path.basename(pdf_path)

    tables_data = extract_tables_from_pdf(pdf_path)
    tables_by_page = {}
    for table in tables_data:
        tables_by_page.setdefault(table['page'], []).append(table['text'])

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(layout=True)
                if text:
                    if page_num in tables_by_page:
                        text += "\n\n" + "\n".join(tables_by_page[page_num])
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "filename": filename,
                            "page": page_num,
                            "file_type": "pdf",
                            "extraction_method": "pdfplumber_advanced"
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
