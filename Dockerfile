FROM nvcr.io/nvidia/tritonserver:20.09-py3
# 20.09

COPY requirements.txt .

RUN apt-get update && apt-get install -y
RUN apt -y install python3-pip
RUN apt-get install -y python3.8
RUN alias python='/usr/bin/python3.8'
RUN pip3 install --upgrade setuptools

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "tritonserver" ]
