import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import json
import markdown
from weasyprint import HTML

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
API_URL = "http://localhost:8000/chat"
API_USERNAME = "admin"
API_PASSWORD = "password"

st.set_page_config(page_title="ResilienceGPT Chatbot", layout="wide")
st.title("🧠 ResilienceGPT — RAG Chatbot (Claude + Qdrant)")


# ------------------------------------------------------------
# CLEAN MARKDOWN
# ------------------------------------------------------------
def clean_llm_markdown(text: str) -> str:
    import re
    text = text.replace("\\n", "\n").replace("\\t", "    ").strip('"')
    text = re.sub(r"\n\s+(\- |\* |\d+\. |#)", r"\n\1", text)
    text = re.sub(r"\n\s*(#{1,6})\s*", r"\n\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------
# PDF EXPORT
# ------------------------------------------------------------
def generate_pdf_from_markdown(markdown_text, output_path):
    html = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])

    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica', sans-serif;
                margin: 2rem;
                line-height: 1.6;
                font-size: 12px;
            }}
            h1, h2, h3, h4 {{
                font-weight: bold;
                margin-top: 1.5rem;
                margin-bottom: .5rem;
            }}
            ul {{
                margin-left: 1.5rem;
                padding-left: 1.2rem;
                list-style-type: disc;
            }}
            ul li {{
                margin-bottom: 4px;
                padding-left: 4px;
            }}
            ol {{
                margin-left: 1.5rem;
                padding-left: 1.2rem;
            }}
            ol li {{
                margin-bottom: 4px;
                padding-left: 4px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 12px;
                border-radius: 8px;
                overflow-x: auto;
                font-size: 11px;
                line-height: 1.4;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 4px;
            }}
            table {{
                border-collapse: collapse;
                margin-top: 1rem;
            }}
            th, td {{
                border: 1px solid #888;
                padding: 6px;
                font-size: 11px;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """

    HTML(string=html_template).write_pdf(output_path)


# ------------------------------------------------------------
# SESSION STATE: CHAT HISTORY
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------------------
for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)


# ------------------------------------------------------------
# CHAT INPUT
# ------------------------------------------------------------
user_input = st.chat_input("Pose une question…")

if user_input:
    # Save user message
    st.session_state.messages.append(("user", user_input))

    # Display immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # SEND TO API
    # --------------------------------------------------------
    try:
        response = requests.post(
            API_URL,
            json={"question": user_input},
            auth=HTTPBasicAuth(API_USERNAME, API_PASSWORD)
        )
    except Exception as e:
        st.error(f"❌ Impossible d’appeler l’API FastAPI : {e}")
        st.stop()

    if response.status_code != 200:
        st.error(f"❌ API error: {response.text}")
        st.stop()

    data = response.json()

    # Clean answer
    clean_answer = clean_llm_markdown(data["answer"])

    # Save assistant message to history
    st.session_state.messages.append(("assistant", clean_answer))

    # Display response
    with st.chat_message("assistant"):
        st.markdown(clean_answer)


# ------------------------------------------------------------
# PDF EXPORT BUTTON (Export entire conversation)
# ------------------------------------------------------------
if st.button("📥 Exporter la conversation en PDF"):
    full_markdown = ""
    for role, msg in st.session_state.messages:
        full_markdown += f"### {role.capitalize()}\n{msg}\n\n"

    out_path = "conversation.pdf"
    generate_pdf_from_markdown(full_markdown, out_path)

    with open(out_path, "rb") as pdf:
        st.download_button(
            "📄 Télécharger le PDF",
            pdf,
            file_name="conversation.pdf",
            mime="application/pdf",
        )
