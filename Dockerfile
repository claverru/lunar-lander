FROM pytorch/pytorch:latest

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /usr/src

CMD bash
