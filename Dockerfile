FROM fedora
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN dnf install -y htop \
	&& dnf install -y wget \
	&& wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& bash Miniconda3-latest-Linux-x86_64.sh -b \
	&& rm Miniconda3-latest-Linux-x86_64.sh

RUN source /root/.bashrc \ 
	&& conda create -n ml

RUN mkdir /root/app/
	
COPY . /root/app/

RUN source activate ml \
	&& pip3 install -r /root/app/requirements.txt # pandas numpy scikit-learn tensorflow seaborn matplotlib flask
	
