import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

load_dotenv()


def get_api_key_for_embedding() -> str:
    api_key = os.getenv("GEMINI_API_KEY_FOR_EMBEDDING")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in your .env file.")
    os.environ["GEMINI_API_KEY_FOR_EMBEDDING"] = api_key
    return api_key

def get_api_key_for_chatbot() -> str:
    api_key = os.getenv("GEMINI_API_KEY_FOR_CHATBOT")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY_FOR_CHATBOT in your .env file.")
    os.environ["GEMINI_API_KEY_FOR_CHATBOT"] = api_key
    return api_key


def extract_video_id(video_input: str) -> Optional[str]:
    video_input = video_input.strip()
    if not video_input:
        return None

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_input):
        return video_input

    parsed_url = urlparse(video_input)
    if parsed_url.netloc and "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if query_params.get("v"):
            return query_params["v"][0]
    if parsed_url.netloc and "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")

    return None


def fetch_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
    except NoTranscriptFound as exc:
        raise ValueError("No transcript found for this video.") from exc
    except TranscriptsDisabled as exc:
        raise ValueError("Transcripts are disabled for this video.") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while fetching transcript: {exc}") from exc

    transcript_parts = [snippet.text for snippet in transcript_list if getattr(snippet, "text", None)]
    return " ".join(transcript_parts)


def build_vector_store(transcript: str, api_key: Optional[str] = None) -> InMemoryVectorStore:
    api_key = get_api_key_for_embedding()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key,
    )

    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=docs)
    return vector_store


def create_chain(video_id: str, history_text: str = "", api_key: Optional[str] = None):
    transcript = fetch_transcript(video_id)
    vector_store = build_vector_store(transcript, api_key=api_key)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=get_api_key_for_chatbot(),
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant who answers the current question using the provided context and conversation history.
        Use the history to maintain continuity and avoid repeating information.
        If you cannot find the answer in the context, say you do not know.

        Conversation history:
        {history}

        Context:
        {context}

        Current question:
        {query}
        """,
        input_variables=["context", "history", "query"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "query": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: history_text),
        }
    )

    return parallel_chain | prompt | llm | StrOutputParser()


def answer_query(video_id: str, query: str, history: Optional[list[dict]] = None, api_key: Optional[str] = None) -> str:
    history_text = "\n".join(
        f"User: {item['user']}\nAssistant: {item['assistant']}" for item in (history or [])
    )
    chain = create_chain(video_id, history_text=history_text, api_key=api_key)
    return chain.invoke(query)


if __name__ == "__main__":
    video_input = input("Enter a YouTube video URL or ID: ").strip()
    video_id = extract_video_id(video_input)
    if not video_id:
        raise SystemExit("Could not parse a YouTube video ID from the input.")

    question = input("Ask a question about the video: ").strip()
    print(answer_query(video_id, question))
