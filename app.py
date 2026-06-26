import streamlit as st

from main import answer_query, extract_video_id, fetch_transcript

st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥")
st.title("YouTube Video Question Answering")
st.caption("Ask questions about a YouTube video using its transcript and Gemini.")

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
    st.session_state.transcript = ""

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
                st.success("Transcript loaded successfully.")

if st.session_state.current_video_id:
    st.subheader(f"Video ID: {st.session_state.current_video_id}")

    preview = st.session_state.transcript
    if len(preview) > 5000:
        preview = preview[:5000] + "..."

    st.text_area("Transcript preview", preview, height=240)

    query = st.text_input("Ask a question about this video")
    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_query(st.session_state.current_video_id, query)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.subheader("Answer")
                    st.write(answer)
