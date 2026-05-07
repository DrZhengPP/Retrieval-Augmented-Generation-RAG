import os, sys, anthropic, chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import CrossEncoder

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-opus-4-7"

def load_documents_from_directory(directory_path: str) -> list:
    documents = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append({"id": filename, "text": f.read()})
        elif filename.endswith(".pdf"):
            pdf = PdfReader(filepath)
            pages = [page.extract_text().strip() for page in pdf.pages]
            pages = [p for p in pages if p]
            documents.append({"id": filename, "text": "\n\n".join(pages)})
    return documents


#In the English language, every 4 words is approximately 1 token. Since most models allow 256 tokens per chunk, we can use a chunk size of 1000 characters to be safe, which is roughly 250 tokens. We also add a small overlap of 5 tokens (20 characters) to help maintain context across chunks.
def split_into_chunks(documents: list, chunk_size=1000, chunk_overlap=5) -> list:
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". "," ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc['text'])):
            chunks.append({"id": f"{doc['id']}_chunk_{i}", "text": chunk_text})
    return chunks

def split_into_tokens(chunks: list, tokens_per_chunk=256, chunk_overlap=0) -> list:
    splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap)
    token_chunks = []
    for chunk in chunks:
        for i, sub in enumerate(splitter.split_text(chunk['text'])):
            token_chunks.append({"id": f"{chunk['id']}_token_{i}", "text": sub})
    return token_chunks

def build_vector_store(token_chunks: list, collection_name='rag-claude') -> chromadb.Collection:
    embedding_fn = SentenceTransformerEmbeddingFunction()
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_fn)
    collection.add(ids=[c['id'] for c in token_chunks], documents=[c['text'] for c in token_chunks])
    return collection


def augment_query_with_hypothetical_answer(query: str) -> str:
    system_prompt = (
        "You are a helpful expert research assistant. When given a question, "
        "produce a concise, plausible example answer exactly as it might appear "
        "verbatim in a formal document such as an annual report, research paper, "
        "or technical specification. Write in declarative, precise prose without "
        "hedging. Do not mention that this answer is hypothetical."
    )
    response = client.messages.create(model=CLAUDE_MODEL,max_tokens=400,thinking={"type": "adaptive"},system=system_prompt,messages=[{"role": "user", "content": query}],)
    hypothetical_answer = next((block.text for block in response.content if block.type == "text"), "")
    print(f"  → Hypothetical answer generated ({len(hypothetical_answer)} chars).")
    return hypothetical_answer

def retrieve_from_vector_store(collection, joint_query: str, n_results=5) -> list:
    results = collection.query(query_texts=joint_query, n_results=n_results)
    retrieved = results['documents'][0]
    return retrieved

def rerank_documents(query: str, documents: list, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2') -> list:
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked]

def generate_final_answer(query: str, context_documents: list) -> str:
    context = "\n\n---\n\n".join(context_documents)
    prompt = (
        f"Using ONLY the context excerpts provided below, answer the following "
        f"question clearly and completely for a non-specialist audience.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT EXCERPTS:\n{context}\n\n"
        f"If the provided context does not contain sufficient information to "
        f"answer the question fully, state that explicitly rather than speculating."
    )
    full_answer_parts = []
    with client.messages.stream(model=CLAUDE_MODEL,max_tokens=1024,thinking={"type": "adaptive"},messages=[{"role": "user", "content": prompt}],) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_answer_parts.append(text)
    print()
    return "".join(full_answer_parts)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python rag_claude.py "Your query here"')  
        sys.exit(1)
    user_query = sys.argv[1]
    documents = load_documents_from_directory("data")
    chunks = split_into_chunks(documents)
    token_chunks = split_into_tokens(chunks)
    collection = build_vector_store(token_chunks)
    hypothetical_answer = augment_query_with_hypothetical_answer(user_query)
    joint_query = f'{user_query} {hypothetical_answer}'
    retrieved_docs = retrieve_from_vector_store(collection, joint_query)
    reranked_docs = rerank_documents(user_query, retrieved_docs)
    print('ANSWER:')
    print(f"{'=' * 50}\n")
    generate_final_answer(user_query, reranked_docs)


### Homework: Modify and update ShobhitGPT with agentic functions similar to what is done in this RAG example. Use both kinds of RAG, hypothetical answer augmentation as well as multiple query augmentation. Improve our AI skills together!
