import streamlit as st

from main import answer_query, extract_video_id, fetch_transcript

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥")
st.title("YouTube Video Chatbot")
st.markdown("Chat with a YouTube video's transcript using Gemini embeddings.")

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
    st.session_state.transcript = ""
    st.session_state.history = []

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

with st.sidebar:
    st.header("Video settings")
    video_input = st.text_input(
        "YouTube video URL or ID",
        placeholder="https://www.youtube.com/watch?v=jGwO_UgTS7I",
    )
    if st.button("Load video"):
        with st.spinner("Loading transcript..."):
            video_id = extract_video_id(video_input)
            if not video_id:
                st.error("Please enter a valid YouTube URL or video ID.")
            else:
                try:
                    transcript = fetch_transcript(video_id)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.session_state.current_video_id = video_id
                    st.session_state.transcript = transcript
                    st.session_state.history = []
                    st.session_state.video_loaded = True
                    st.success("Video loaded successfully.")

    if st.session_state.video_loaded:
        if st.button("Clear chat"):
            st.session_state.history = []
            st.success("Chat history cleared.")

if st.session_state.current_video_id:
    st.info(f"Loaded video ID: {st.session_state.current_video_id}")

    with st.expander("Transcript preview", expanded=False):
        preview = st.session_state.transcript
        if len(preview) > 10000:
            preview = preview[:10000] + "..."
        st.write(preview)

    if st.session_state.history:
        for turn in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(turn["user"])
            with st.chat_message("assistant"):
                st.markdown(turn["assistant"])

    user_prompt = st.chat_input("Ask a question about the video...")
    if user_prompt:
        if not user_prompt.strip():
            st.warning("Please enter a question.")
        else:
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_placeholder.markdown("Generating answer...")

            try:
                answer = answer_query(
                    st.session_state.current_video_id,
                    user_prompt,
                    history=st.session_state.history,
                )
            except Exception as exc:
                response_placeholder.markdown(f"Sorry, something went wrong: {exc}")
            else:
                response_placeholder.markdown(answer)
                st.session_state.history.append(
                    {
                        "user": user_prompt,
                        "assistant": answer,
                    }
                )
else:
    st.warning("Load a YouTube video in the sidebar to start chatting.")
