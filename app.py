import os
import base64
import json
from io import BytesIO

import streamlit as st
from streamlit.components.v1 import html as st_html

import anthropic
from PIL import Image


# ---------- Anthropic client helper ----------

def get_claude_client() -> anthropic.Anthropic:
    api_key = None

    # 1) Try Streamlit secrets
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    # 2) Fallback to environment variable
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Claude API key not found. Please set ANTHROPIC_API_KEY in "
            "Streamlit secrets or as an environment variable."
        )

    return anthropic.Anthropic(api_key=api_key)


# ---------- Image helper: normalize to PNG ----------

def to_png_bytes(raw_bytes: bytes) -> bytes:
    """
    Open the uploaded image and re-encode it as PNG bytes.
    This avoids media-type mismatches like 'image/jpeg' vs webp/other.
    """
    img = Image.open(BytesIO(raw_bytes))
    rgb = img.convert("RGBA") if img.mode in ("P", "LA") else img.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Claude helper ----------

def describe_image_with_claude(image_bytes: bytes, max_words: int) -> str:
    """
    Send an image + instructions to Claude and get back a markdown description
    following your #Approach.
    """

    # Hard cap at 300 words regardless of slider
    max_words = min(max_words, 300)

    # Normalize image to PNG to avoid media-type mismatch
    png_bytes = to_png_bytes(image_bytes)
    b64_image = base64.b64encode(png_bytes).decode("utf-8")

    client = get_claude_client()

    user_prompt = f"""
You will see an image. Follow this approach strictly:

#Approach

1. Begin your answer with a line in this exact format: **Picture Type:** <short type, e.g. "Car picture", "Baby picture", "Food picture">.
2. After that, write a short 1–2 sentence explanation of the overall picture.
3. Then explain the picture in **Markdown format** using descriptive language.
4. Keep the entire response under **{max_words} words**, and never exceed **300 words** in any case. HARD LIMIT.
5. Format the description so it is easily scannable (for example, bullets with bold labels, short lines).
6. Use **only** what is visible in the provided image. Do not invent or guess hidden details.
7. Output **only** the description itself (starting with the “Picture Type” line). Do not repeat these instructions or add extra commentary.
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=600,
        system=(
            "You are an assistant that writes short, structured, accessible image "
            "descriptions for people who cannot see the image. "
            "Always start with a 'Picture Type' line and respect the word limit."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",  # we now always send PNG
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )

    text_chunks = [
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ]
    description = "\n\n".join(text_chunks)
    return enforce_word_limit(description, max_words)


def enforce_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


# ---------- Streamlit UI ----------

st.set_page_config(page_title="The Describer", layout="centered")

if "description" not in st.session_state:
    st.session_state.description = ""

# Load cat image for header
CAT_PATH = "cool_cat.jpg"   # make sure this file is next to app.py
with open(CAT_PATH, "rb") as f:
    cat_b64 = base64.b64encode(f.read()).decode("utf-8")

# ---------- Styles ----------
st.markdown(
    f"""
    <style>
    /* Completely hide Streamlit header + pill */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}

    .stApp {{
        background: radial-gradient(circle at top, #ffe9c7 0, #f5f7ff 45%, #f0f0f0 100%);
    }}

    .main .block-container {{
        padding-top: 0.5rem;
        padding-bottom: 2.5rem;
    }}

    .describer-card {{
        max-width: 460px;
        margin: 0 auto 0 auto;
        padding: 0.75rem 1.5rem 1.5rem 1.5rem;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
    }}

    .describer-title {{
        text-align: center;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.4rem;
        margin-bottom: 0.3rem;
    }}

    .describer-subtitle {{
        text-align: center;
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 1rem;
    }}

    /* Center cat image */
    .hero-cat-wrap {{
        display: flex;
        justify-content: center;
        margin-top: 0.3rem;
        margin-bottom: 0.4rem;
    }}

    .hero-cat-img {{
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
        width: 260px;
        height: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="describer-card">', unsafe_allow_html=True)

# ---------- Center Cat ----------
st.markdown(
    f"""
    <div class="hero-cat-wrap">
        <img class="hero-cat-img" src="data:image/jpeg;base64,{cat_b64}" />
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Title ----------
st.markdown('<div class="describer-title">The Describer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="describer-subtitle">Upload an image and get a concise AI-generated description.</div>',
    unsafe_allow_html=True,
)

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Drop image here or click to upload",
    type=["png", "jpg", "jpeg", "gif", "webp"],
    label_visibility="collapsed",
)

if uploaded_file:
    st.image(uploaded_file)

# ---------- Slider ----------
max_words = st.slider(
    "Description Length (words)",
    min_value=1,
    max_value=300,     # max 300 now
    value=100,         # default 100
    help="Claude will try to stay under this word limit (absolute maximum 300).",
)

col1, col2 = st.columns([1, 1])
with col1:
    describe_clicked = st.button("Describe", use_container_width=True, type="primary")
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.description = ""

if describe_clicked:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Asking Claude to describe your image..."):
            try:
                st.session_state.description = describe_image_with_claude(
                    uploaded_file.getvalue(),
                    max_words,
                ).strip()
            except Exception as e:
                st.error(f"Something went wrong talking to Claude: {e}")

st.markdown("### Description")

if st.session_state.description:
    st.markdown(st.session_state.description)
else:
    st.write("_No description yet. Upload an image and click **Describe**._")

# ---------- Copy Controls ----------
col_copy, col_clear2 = st.columns([1, 1])
with col_copy:
    copy_clicked = st.button("Copy", use_container_width=True)
with col_clear2:
    clear2_clicked = st.button("Clear text", use_container_width=True)

if clear2_clicked:
    st.session_state.description = ""

if copy_clicked and st.session_state.description:
    st_html(
        f"""
        <script>
        navigator.clipboard.writeText({json.dumps(st.session_state.description)});
        </script>
        """,
        height=0,
    )
    st.success("Description copied to clipboard!")

st.markdown("</div>", unsafe_allow_html=True)
