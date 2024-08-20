from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
import streamlit as st
import os
import tempfile

# Setup OpenAI API
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(temperature=0)

# Function to summarize PDFs from a folder


def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())

        loader = PyPDFLoader(temp_path)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        summaries.append(summary)

        # Delete the temporary file
        os.remove(temp_path)

    return summaries


# Set up the page configuration with orange theme colors
st.set_page_config(page_title="Incenteev Summarizer",
                   page_icon="💡", layout="centered")

# Display the company logo
st.image("incenteev.webp", width=200)

# Set the title with an orange theme
st.markdown(
    "<h1 style='text-align: center;'>🧐 Incenteev Smart Summarizer</h1>",
    unsafe_allow_html=True,
)

# Allow user to upload PDF files with a themed uploader
pdf_files = st.file_uploader(
    "You can upload multipe PDF files and you'll get a summary for each one📚", type="pdf", accept_multiple_files=True
)

# If PDFs are uploaded, display the summary generation button with a themed button
if pdf_files:
    if st.button("Generate Summary 🔥", help="Click to generate summaries for your PDFs"):
        st.write("Here's the summary, enjoy 😉:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"### Summary for PDF {i+1}:")
            st.write(summary)

# Footer or additional information can also follow the theme
st.markdown(
    "<p style='text-align: center;'>Powered by Incenteev</p>",
    unsafe_allow_html=True,
)
