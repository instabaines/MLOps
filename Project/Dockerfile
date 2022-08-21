FROM python:3.9.0-slim

RUN pip install -U pip

WORKDIR /app
COPY requirement.txt .


RUN pip install -r requirement.txt

RUN pip3 install evidently

COPY ["webservice/preditcion service/predict.py", "webservice/predict//model.pkl" ,"/app/"]




EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"  ]