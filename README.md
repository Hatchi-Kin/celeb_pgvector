- Pour lancer la bdd

```bash
docker-compose up -d
```

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

- Pour remplir la bdd,  :anger: run all cell dans bordel/load_db_with_embeddings.ipynb   :warning:

```bash
python app.py
```
- puis se rendre sur http://127.0.0.1:5000/

- Pour stop tous les containers en cours
```bash
sudo docker stop $(sudo docker ps -a -q) 
```
- Pour Effacer touts les conteneurs, images et volumes

```bash
sudo docker system prune -a --volumes 
```
## repos de references

https://github.com/pgvector/pgvector


https://github.com/pgvector/pgvector-python