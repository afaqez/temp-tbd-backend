import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import fitz  # PyMuPDF for handling PDF files

load_dotenv()

# Define the directory containing the PDF files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "dataset")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Dataset directory: {dataset_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"The directory {dataset_dir} does not exist. Please check the path."
        )

    # List all PDF files in the directory
    dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith(".pdf")]
    print(f"Dataset files: {dataset_files}")

    # Read the text content from each file and store it with metadata
    documents = []
    for dataset_file in dataset_files:
        file_path = os.path.join(dataset_dir, dataset_file)
        doc = fitz.open(file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text('text')
            if text.strip():  # Ensure the text is not empty
                documents.append(Document(page_content=text, metadata={"source": dataset_file, "page": page_num}))
                print(f"Extracted text from {dataset_file}, page {page_num}:")
                print(text[:200])  # Print the first 200 characters of the extracted text for debugging
            else:
                print(f"Empty text on page {page_num} of {dataset_file}")

    if not documents:
        raise ValueError("No documents were loaded. Please check your dataset.")

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    if len(docs) == 0:
        raise ValueError("No document chunks were created. Please check your text splitting logic.")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
