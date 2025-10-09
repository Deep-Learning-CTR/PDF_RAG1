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
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Suppress Camelot image-based page warnings
warnings.filterwarnings('ignore', message='.*is image-based, camelot only works on text-based pages.*')

# Set Tesseract path for Windows (adjust if installed elsewhere)
if os.name == 'nt':  # Windows
    tesseract_path = r'C:\Users\charb\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Initialize Groq client for vision models
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def describe_image_with_vision(image_pil, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """Use Groq's vision model to describe image content"""
    try:
        # Convert PIL image to base64
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Call Groq vision API
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Include what objects, people, charts, diagrams, or visual elements are present. Be concise but informative."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=300
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error describing image with vision model: {e}")
        return None

def extract_all_from_pdf_page(page, page_num, use_camelot_tables=None, use_vision=True):
    """Extract text, tables, and images from a single PDF page"""
    extracted_data = {
        'text': '',
        'tables': [],
        'ocr_text': [],
        'image_descriptions': []
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

    # Extract text from images using OCR and Vision models
    try:
        images = page.images

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

                # Try OCR first
                ocr_text = pytesseract.image_to_string(pil_img)
                if ocr_text.strip():
                    extracted_data['ocr_text'].append(f"\n[IMAGE {i+1} OCR TEXT]\n{ocr_text.strip()}\n[END IMAGE {i+1}]\n")

                # If no text found via OCR and vision is enabled, use vision model
                elif use_vision:
                    description = describe_image_with_vision(pil_img)
                    if description:
                        extracted_data['image_descriptions'].append(f"\n[IMAGE {i+1} DESCRIPTION]\n{description}\n[END IMAGE {i+1}]\n")

            except Exception as e:
                print(f"Error extracting from image {i+1} on page {page_num}: {e}")
    except Exception as e:
        print(f"Error accessing images on page {page_num}: {e}")

    # Add Camelot tables if available
    if use_camelot_tables and page_num in use_camelot_tables:
        extracted_data['tables'].extend(use_camelot_tables[page_num])

    return extracted_data


def extract_text_from_pdf_advanced(pdf_path, use_vision=True):
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
                page_data = extract_all_from_pdf_page(page, page_num, camelot_tables, use_vision)

                # Combine all extracted content
                combined_text = page_data['text'] or ""
                if page_data['ocr_text']:
                    combined_text += "\n\n" + "\n".join(page_data['ocr_text'])
                if page_data['image_descriptions']:
                    combined_text += "\n\n" + "\n".join(page_data['image_descriptions'])
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
                            "has_image_descriptions": bool(page_data['image_descriptions']),
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


def extract_text_from_multiple_files(file_paths, use_vision=True):
    """Extract text from multiple PDF and Excel files"""
    all_documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            all_documents.extend(extract_text_from_pdf_advanced(file_path, use_vision))
        elif ext in [".xlsx", ".xls"]:
            all_documents.extend(extract_text_from_excel(file_path))
    return all_documents


def extract_from_standalone_image(image_path, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """Extract description from a standalone image file using vision model"""
    documents = []
    filename = os.path.basename(image_path)

    try:
        # Open image
        pil_img = Image.open(image_path)

        # Try OCR first
        ocr_text = pytesseract.image_to_string(pil_img)

        # Get vision description
        vision_description = describe_image_with_vision(pil_img, model)

        # Combine OCR and vision results
        combined_text = ""
        if ocr_text.strip():
            combined_text += f"[OCR TEXT FROM IMAGE]\n{ocr_text.strip()}\n[END OCR TEXT]\n\n"

        if vision_description:
            combined_text += f"[IMAGE DESCRIPTION]\n{vision_description}\n[END IMAGE DESCRIPTION]"

        if combined_text.strip():
            doc = Document(
                page_content=combined_text,
                metadata={
                    "source": image_path,
                    "filename": filename,
                    "file_type": "image",
                    "extraction_method": "vision_ocr_combined",
                    "has_ocr": bool(ocr_text.strip()),
                    "has_vision_description": bool(vision_description)
                }
            )
            documents.append(doc)

    except Exception as e:
        print(f"Error extracting from image {filename}: {e}")

    return documents

def split_chunk_overlap(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents with special handling for tables"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    return splitter.split_documents(documents)
