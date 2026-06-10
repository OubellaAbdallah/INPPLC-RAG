import sys
import logging
import warnings
from typing import List

from langchain_core.messages import HumanMessage, AIMessage

from src.config.settings import get_settings
from src.generation.classifier import QueryClass, classify_query
from src.generation.generator import ResponseGenerator, get_llm
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.vectorstore import get_vectorstore
from src.retrieval.reranker import DocumentReranker
from src.retrieval.repacker import DocumentRepacker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)

def main():
    print("=========================================")
    print("Initializing INPPLC RAG Terminal Client...")
    print("=========================================")
    
    settings = get_settings()
    
    print("[1/3] Loading LLM model (Ollama)...")
    try:
        llm = get_llm(settings)
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return
    
    print("[2/3] Loading Vector Store (ChromaDB)...")
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        print(f"Error loading Vector Store: {e}")
        return
        
    print("[3/3] Loading Sparse Index (TF-IDF)...")
    tfidf_index = TfidfIndex()
    try:
        tfidf_index.load()
    except Exception as e:
        print(f"Warning: Could not load TF-IDF index: {e}")
        
    generator = ResponseGenerator(llm=llm, settings=settings)
    chat_history = []
    
    print("\nInitialization Complete!")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("=========================================\n")
    
    while True:
        try:
            query = input("User> ")
            if query.strip().lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if not query.strip():
                continue
                
            chat_history.append(HumanMessage(content=query))
            
            # 1. Classify query
            print("  [Classifying query...]", end="\r")
            try:
                query_class = classify_query(query, llm, settings)
                print(f"  [Query classified as: {query_class.name}]     ")
            except Exception as e:
                print(f"  [Query classification failed, defaulting to RAG_REQUIRED]")
                query_class = QueryClass.RAG_REQUIRED
            
            context_docs = []
            
            # 2. Retrieve Context (if needed)
            if query_class == QueryClass.RAG_REQUIRED:
                print("  [Retrieving context...]", end="\r")
                context_docs = hybrid_retrieve(
                    query=query,
                    vectorstore=vectorstore,
                    tfidf_index=tfidf_index,
                    settings=settings,
                )
                
                # Reranking
                if settings.reranking_enabled and context_docs:
                    print("  [Reranking documents...] ", end="\r")
                    reranker = DocumentReranker(settings)
                    context_docs = reranker.rerank(query, context_docs)
                    
                # Repacking
                if settings.repacking_enabled and context_docs:
                    print("  [Repacking context...]   ", end="\r")
                    repacker = DocumentRepacker(settings)
                    context_docs = repacker.repack_by_similarity(context_docs)
                
                if context_docs:
                    print(f"  [Found {len(context_docs[:10])} relevant context chunks]")
                else:
                    print("  [No relevant context found in database]   ")
            
            # 3. Generate Response
            print("Assistant> ", end="", flush=True)
            response_stream = generator.generate_stream(
                query=query,
                context_documents=context_docs[:10] if context_docs else None,
                conversation_history=chat_history,
                query_class=query_class
            )
            
            full_response = ""
            for chunk in response_stream:
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            
            chat_history.append(AIMessage(content=full_response))
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError processing query: {e}\n")

if __name__ == "__main__":
    main()
