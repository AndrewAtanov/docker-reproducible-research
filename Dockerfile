# Base image
FROM python:3.6

RUN pip install configargparse numpy matplotlib sklearn pyyaml torch torchvision

RUN pip install filelock pandas
RUN pip install tabulate
RUN pip install seaborn

RUN apt-get update && apt-get install -y texlive

ADD run.sh ./
ADD code ./code
ADD data ./data
ADD latex ./latex

VOLUME /example/results

RUN chmod +x run.sh

CMD ./run.sh


