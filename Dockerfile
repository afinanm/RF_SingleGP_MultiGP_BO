FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''" ]
