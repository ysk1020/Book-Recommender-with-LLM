import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')

books["authors"] = books["authors"].fillna("Unknown author")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.png",
    books["large_thumbnail"],
)


raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)

documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(
    documents,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)

def retrieve_semantic_recommendations(
        query:str,
    category:str = None,
    tone:str = None,
    initial_top_k:int=50,
    final_top_k:int=16,
    ) -> pd.DataFrame:
    recs=db_books.similarity_search(query, k=initial_top_k)    
    book_list=[int(recs.page_content.strip('"').split(' ')[0]) for recs in recs ]
    books_recs=books[books["isbn13"].isin(book_list)].head(initial_top_k)


    if category!= "All":
        books_recs=books_recs[books_recs["simple_categories"]==category].head(final_top_k)
    else:
        books_recs=books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone== "Surprise":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone== "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone== "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone== "Anger":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone== "Disgust":
        books_recs.sort_values(by="disgust", ascending=False, inplace=True)


    return books_recs

def recommend_books(query: str,  category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    details = []
    for _, row in recommendations.iterrows():
        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            authors_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            authors_str = ", ".join(author_split[:-1]) + f", and {author_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}"
        results.append((row["large_thumbnail"], caption))
        details.append(
            f"### {row['title']}\n"
            f"**by {authors_str}** · *{row['simple_categories']}*\n\n"
            f"{row['description']}"
        )
    return results, details, ""

def show_details(details: list, evt: gr.SelectData) -> str:
    return details[evt.index]


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"]+["Happy", "Surprise", "Sad", "Suspenseful", "Anger", "Disgust"]

with gr.Blocks() as dashboard:
    gr.Markdown("<h1 style='text-align: center;'>Semantic Book Recommender</h1>")
    with gr.Row():
        query_input = gr.Textbox(
            label="Please enter a description of a book",
            placeholder="e.g., A story about forgiveness",
            scale=3)
        category_input = gr.Dropdown(
            label="Select a category",
            choices=categories,
            value="All",
            scale=1)
        tone_input = gr.Dropdown(
            label="Select a tone",
            choices=tones,
            value="All",
            scale=1)
        submit_button = gr.Button("Get Recommendations", variant="primary", scale=1)

    gr.Markdown("<h2 style='text-align: center;'>Recommended Books</h2>")
    output_gallery = gr.Gallery(show_label=False, columns=8, rows=2, height="auto", object_fit="contain")
    details_state = gr.State([])
    details_box = gr.Markdown()

    submit_button.click(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=[output_gallery, details_state, details_box]
    )
    query_input.submit(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=[output_gallery, details_state, details_box]
    )
    output_gallery.select(
        fn=show_details,
        inputs=[details_state],
        outputs=[details_box]
    )

if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Soft())