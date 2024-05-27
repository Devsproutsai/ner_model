FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt 

COPY . /code/

EXPOSE 8000


CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000", "--workers=5"]