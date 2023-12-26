- Pour lancer la bdd

```bash
docker-compose up -d
```

- Pour remplir la bdd, run all cell dans bordel/load_db_with_embeddings.ipynb

- Pour lancer l'app
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
python app.py
```
- puis se rendre sur http://127.0.0.1:5000/


## repos de references

https://github.com/pgvector/pgvector


https://github.com/pgvector/pgvector-python