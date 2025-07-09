import streamlit as st
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


from aws_upload import upload_to_s3


st.title("Bangla Enterprise AI Chatbot (OCR + AWS + Oracle)")

st.markdown("""
**Features:**  
- Upload printed/handwritten Bangla image or text  
- Extract text with OCR  
- Retrieval-Augmented QnA chatbot (Bangla & English)  
- Upload extracted data to AWS S3  

""")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = load_model()

uploaded_file = st.file_uploader(
    "Upload Bangla document (image or .txt)", type=['png', 'jpg', 'jpeg', 'txt']
)
ocr_text = ""
doc_texts = []
embeddings = []

if uploaded_file:
    if uploaded_file.type.startswith('image'):
        img = Image.open(uploaded_file)
        ocr_text = pytesseract.image_to_string(img, lang='ben+eng')
        st.text_area("Extracted OCR Text", ocr_text, height=200)
        with open("ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)
    else:
        ocr_text = uploaded_file.read().decode("utf-8")
        st.text_area("Document Text", ocr_text, height=200)
        with open("ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)

    # Chunk for retrieval
    doc_chunks = [ocr_text[i:i+200] for i in range(0, len(ocr_text), 200) if ocr_text.strip()]
    doc_texts = doc_chunks
    embeddings = model.encode(doc_chunks) if doc_chunks else []

    st.success("Document processed! Now try QnA or integrations.")

# QnA Chatbot section
question = st.text_input("Ask a question about the document (Bangla/English):")
if question and embeddings is not None and len(embeddings) > 0:
    q_emb = model.encode([question])[0]
    sims = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    top_idx = np.argmax(sims)
    st.markdown(f"**Answer:** {doc_texts[top_idx]}")

# AWS S3 Upload (REAL)
with st.expander("Upload OCR result to AWS S3 (real)"):
    st.markdown("**Enter your AWS credentials (never share them, keep secure!)**")
    bucket = st.text_input("AWS S3 Bucket Name", value="", key="bucket")
    aws_access_key = st.text_input("AWS Access Key ID", type="password", key="aws_access")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password", key="aws_secret")
    region = st.text_input("AWS Region (e.g. ap-south-1)", value="ap-south-1", key="region")
    if st.button("Upload to S3"):
        if bucket and aws_access_key and aws_secret_key and region:
            success, msg = upload_to_s3(
                "ocr_result.txt", bucket,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region=region
            )
            st.info(msg if success else f"Failed: {msg}")
        else:
            st.warning("Please enter all AWS credentials and bucket info.")

# Oracle REST API Push (REAL)


st.caption("Demo: Bangla OCR + Chatbot + Real Cloud Integration | By Nasif Ahmed Nafi")
