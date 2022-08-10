FROM eidos-service.di.unito.it/eidos-base-pytorch:1.11.0

COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src


#RUN mkdir /scratch
#RUN chmod 775 /scratch
#RUN chown -R :1337 /scratch

WORKDIR /src

COPY src/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install wandb --upgrade

ENTRYPOINT ["python3"]