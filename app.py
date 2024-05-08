from flask import Flask, request, jsonify
import pandas as pd
import csv
import tempfile
import os
import pickle
import redis
from dotenv import load_dotenv

from pandasai import SmartDataframe
from pandasai.llm import OpenAI

load_dotenv()
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# Configuração do Redis
redis_conn = redis.StrictRedis(
    host=os.getenv("AZURE_REDIS_HOSTNAME"),
    port=6380,
    db=0,
    password=os.getenv("AZURE_REDIS_PASSWORD"),
    ssl=True,
)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "txt"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_delimiter(file_path):
    with open(file_path, "r") as f:
        sample = f.read(1024)
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # Verifica se o arquivo está presente na requisição
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo encontrado"})

        file = request.files["file"]

        # Verifica se o arquivo tem um nome
        if file.filename == "":
            return jsonify({"error": "Nome de arquivo vazio"})

        # Verifica se a extensão do arquivo é permitida
        if not allowed_file(file.filename):
            return jsonify({"error": "Tipo de arquivo não permitido"})

        # Verifica o tamanho do arquivo (limite de 5 MB)
        file_size_mb = request.content_length / (1024 * 1024)
        if file_size_mb > 5:
            return jsonify({"error": "O tamanho do arquivo excede 5 MB"})

        # Salva o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            file_path = temp_file.name

        # Detecta o delimitador do arquivo CSV
        delimiter = ","
        if file.filename.endswith(".csv"):
            delimiter = detect_delimiter(file_path)

        # Lê o arquivo e converte para DataFrame pandas
        if file.filename.endswith((".csv", ".txt")):
            df_pandas = pd.read_csv(file_path, delimiter=delimiter)
        elif file.filename.endswith(".xlsx"):
            df_pandas = pd.read_excel(file_path)

        # Excluir o arquivo temporário após transformá-lo em DataFrame
        os.unlink(file_path)

        # Cache do DataFrame
        redis_conn.set("df_cache", pickle.dumps(df_pandas))

        # Retorna o DataFrame como JSON
        return jsonify({"success": "Arquivo importado"})

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/question", methods=["POST"])
def send_question():
    req = request.get_json()
    cached_df = redis_conn.get("df_cache")
    if cached_df:
        df_pandas = pickle.loads(cached_df)
        df_smart = SmartDataframe(df_pandas, config={"llm": llm})
        answer = df_smart.chat(req["question"])
        return jsonify(answer)
    else:
        return jsonify(
            {
                "error": "DataFrame não encontrado. Por favor, faça o upload do arquivo primeiro."
            }
        )


app.run()
