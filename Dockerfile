FROM python

WORKDIR /reads-flask-app

COPY requirement.txt requirement.txt
RUN pip install -r requirement.txt

COPY . .

CMD [ "python", "wsgi.py"]