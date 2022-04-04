FROM ubuntu:bionic

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.10.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install torchvision==0.11.3+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install torchaudio==0.10.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install pandas numpy scipy 
RUN pip3 install pillow matplotlib
RUN pip3 install sklearn
RUN pip3 install tqdm
RUN pip3 install ipython
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3-tk
RUN pip3 install vincenty 

# todo: pip je ovdje povezan samo s python3 (python3.6), a nije s python3.8