FROM python:3

WORKDIR /mlapp

COPY requirements.txt ./
COPY model/Rental_prediction_model.pkl ./model/Rental_prediction_model.pkl
COPY app.py ./app.py

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [ "python", "app.py" ]

