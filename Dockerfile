FROM python:3

LABEL maintainer="matthew@clinetechnologysolutions.com"

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "./prediction_web_service.py" ]

