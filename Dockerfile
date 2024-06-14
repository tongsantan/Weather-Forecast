FROM python:3.8
COPY . /app
WORKDIR /app
RUN  pip install -r requirements.txt
EXPOSE XXXX 
CMD ["gunicorn", "--bind", "0.0.0.0:XXXX", "app:app"]