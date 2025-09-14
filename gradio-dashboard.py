import gradio as gr
import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load books dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "no_cover.jpg",
    books["large_thumbnail"]
)

# Load documents for embeddings
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())

# --- Recommender Functions ---
def retrieve_semantic_recommensations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 12
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    book_list = [int(rec.page_content.strip('"').split(" ")[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(book_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommensations(query, category, tone)
    cards = ""

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split(".")
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # check if cover exists
        cover_img = (
            row['large_thumbnail']
            if pd.notna(row['large_thumbnail']) and str(row['large_thumbnail']).strip() != ""
            else "no_cover.jpg"
        )

        # append each card
        cards += f"""
        <div style='border:1px solid #ddd; padding:15px; border-radius:15px; 
                    margin:10px; width:250px; display:inline-block; 
                    vertical-align:top; box-shadow:3px 3px 10px rgba(0,0,0,0.15); 
                    text-align:center; background-color:#fdfdfd;'>
            <img src="{cover_img}" alt="Book Cover" 
                style="width:160px; height:auto; border-radius:8px;"/><br>
            <h3 style="margin:10px 0; color:#333;">📖 {row['title']}</h3>
            <p><b>Author:</b> {authors_str}</p>
            <p style="font-size:14px; color:#555;">{truncated_description}</p>
            <span style="font-size:12px; background:#eef; padding:4px 8px; 
                        border-radius:8px; color:#336;">{row['simple_categories']}</span>
        </div>
        """

    return cards

# --- Dashboard UI ---
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme= gr.themes.Monochrome()) as dashboard:
    gr.HTML(
        """<div style='text-align:center; padding:20px;'>
               <h1 style='color:#3b5998;'>📚 Semantic Book Recommender</h1>
               <p style='font-size:16px; color:#555;'>Discover books that match your <b>interests</b> and <b>emotions</b>.</p>
           </div>"""
    )

    with gr.Tab("🔎 Book Recommendations"):
        with gr.Row():
            user_query = gr.Textbox(
                label="Enter a description:",
                placeholder="e.g., A story about forgiveness and family bonds",
                lines=2,
                scale=3
            )
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Category:",
                value="All",
                scale=1
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Emotional Tone:",
                value="All",
                scale=1
            )

        with gr.Row():
            gr.HTML("<div style='text-align:center; width:100%;'>")
            submit_button = gr.Button("✨ Find Books", variant="primary", scale=1)
            gr.HTML("</div>")

        rec_output = gr.HTML(label="Recommended Books")
        submit_button.click(fn=recommend_books,
                            inputs=[user_query, category_dropdown, tone_dropdown],
                            outputs=rec_output)

if __name__ == "__main__":
    dashboard.launch(share=True)
