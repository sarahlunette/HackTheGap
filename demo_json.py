import json
import streamlit as st
import markdown
from weasyprint import HTML

st.set_page_config(page_title="ChatGPT JSON Viewer", layout="wide")
st.title("📄 JSON Viewer — ChatGPT Style + Perfect Lists + Styled PDF Export")


# --- CLEANING FUNCTION ---
def clean_llm_markdown(text: str) -> str:
    import re
    text = text.replace("\\n", "\n").replace("\\t", "    ").strip('"')
    text = re.sub(r"\n\s+(\- |\* |\d+\. |#)", r"\n\1", text)
    text = re.sub(r"\n\s*(#{1,6})\s*", r"\n\1 ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- PDF GENERATOR (with improved list styling) ---
def generate_pdf_from_markdown(markdown_text, output_path):
    # Convert Markdown to HTML
    html = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])

    # Styled HTML template
    html_template = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Helvetica', sans-serif;
                margin: 2rem;
                line-height: 1.6;
                font-size: 12px; /* Smaller font but still readable */
            }}

            h1, h2, h3, h4 {{
                color: #222;
                font-weight: bold;
                margin-top: 1.5rem;
                margin-bottom: .5rem;
            }}

            /* --- PERFECT LIST FORMAT --- */

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

            /* Nested lists */
            ul ul, ol ol {{
                margin-left: 1rem;
                padding-left: 1rem;
                list-style-type: circle;
            }}

            /* Code blocks */
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

            /* Tables */
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


# --- STREAMLIT UI ---
uploaded_file = st.file_uploader("Upload JSON", type=["json"])

if uploaded_file:
    data = json.load(uploaded_file)
    raw_text = data.get("answer", json.dumps(data, ensure_ascii=False))
    clean_text = clean_llm_markdown(raw_text)

    st.markdown("### 👁️ Cleaned Output (Markdown Rendering)")
    st.markdown(clean_text)

    if st.button("📥 Export Styled PDF"):
        pdf_path = "report.pdf"
        generate_pdf_from_markdown(clean_text, pdf_path)

        with open(pdf_path, "rb") as pdf:
            st.download_button(
                "Download PDF",
                pdf,
                file_name="report.pdf",
                mime="application/pdf"
            )