from flask import Flask, render_template, request
from website.models import CelebEmbedding, session
from website.process_pipeline import ImageProcessor
from keras_facenet import FaceNet
from sqlalchemy import Column, Integer, String, create_engine, text, select
from sqlalchemy.sql import func

app = Flask(__name__)
processor = ImageProcessor()
embedder = FaceNet()


output_directory = "static/processed_images"


@app.route('/')
def index():
    random_celeb_embedding = session.query(CelebEmbedding).order_by(func.random()).first()
    query = select(CelebEmbedding).order_by(CelebEmbedding.embedding.l2_distance(random_celeb_embedding.embedding)).limit(5)
    results = session.scalars(query).all()
    top_5 = []
    for result in results:
        top_5.append(result.celebname)
    return render_template('index.html')


@app.route('/matches', methods=['GET', 'POST'])
def matches():
    if request.method == 'POST':
        uploaded_image = request.files.get("image")
        if not uploaded_image:
            return render_template("index.html", prediction_failed="Pas d'image envoyée ... Réessaie encore boulet va")
        uploaded_image_processed_path = processor.process_user_uploaded_image(uploaded_image, processor, output_directory)
        if uploaded_image_processed_path is None:
            return render_template("index.html", prediction_failed='MTCNN could not detect a face. Please try another image.')
        df_embeddings = processor.extract_user_uploaded_embeddings(uploaded_image_processed_path, embedder)
        # Combine the 512 columns into a single list
        embedding = df_embeddings.values.tolist()[0]
        # Calculate the L2 distance
        query = select(CelebEmbedding).order_by(CelebEmbedding.embedding.l2_distance(embedding)).limit(5)
        results = session.scalars(query).all()
        top_5 = []
        for result in results:
            top_5.append({
                'celebname': result.celebname,
                'filepath': result.filepath,
                'target': result.target
            })

        return render_template('index.html', top_5=top_5)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)