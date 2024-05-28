FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt 

RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader words && \
    python -c "import nltk; nltk.download('stopwords')" && \
    python -c "import nltk; nltk.download('punkt')" && \
    python -c "import nltk; nltk.download('averaged_perceptron_tagger')" && \
    python -c "import nltk; nltk.download('wordnet')" && \
    python -c "import nltk; nltk.download('maxent_ne_chunker')" && \
    python -c "import nltk; nltk.download('words')" && \
    python -c "import nltk; nltk.download('omw')" && \
    python -c "import nltk; nltk.download('universal_tagset')" && \
    python -c "import nltk; nltk.download('tagsets')"

COPY . /code/

EXPOSE 8000


CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--workers=5"]