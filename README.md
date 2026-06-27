# RAG-Based YouTube Chatbot

A Python application that lets you chat with the transcript of a YouTube video using a Retrieval-Augmented Generation (RAG) pipeline with Google Gemini and LangChain.

## Overview

This project uses RAG to answer questions about a YouTube video by:
- extracting a YouTube video ID from a video URL or direct ID
- fetching the video transcript
- splitting the transcript into chunks
- creating embeddings and storing them in memory
- retrieving the most relevant chunks and using them with Gemini to generate grounded answers

It includes both:
- a command-line interface in main.py
- a Streamlit web app in app.py

## Project Structure

- main.py - core logic for transcript fetching, embedding creation, retrieval, and answer generation
- app.py - Streamlit interface for loading a video and chatting with it
- requirements.txt - Python dependencies

## Prerequisites

- Python 3.9+
- A Google Gemini API key
- Internet access to fetch YouTube transcripts

## Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a .env file in the project root with your Gemini API keys:

```env
GEMINI_API_KEY_FOR_EMBEDDING=your_embedding_api_key
GEMINI_API_KEY_FOR_CHATBOT=your_chat_api_key
```

## Running the App

### Streamlit web app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

### Command-line version

```bash
python main.py
```

You will be prompted to enter:
- a YouTube video URL or ID
- a question about the video

## Notes

- Some videos may not have transcripts available, in which case the app will show an error.
- Transcript availability depends on the video's settings and YouTube metadata.

## Dependencies

The project uses:
- streamlit
- langchain-google-genai
- youtube-transcript-api
- langchain-text-splitters
- langchain-core
- python-dotenv
