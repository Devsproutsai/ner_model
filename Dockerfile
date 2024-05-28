FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt 

RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader words && \
    python -c "import nltk; nltk.download('stopwords')"

COPY . /code/

EXPOSE 8000


CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--workers=5"]