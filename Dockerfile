# Use the official Python image as the base
FROM python:3.10

RUN mkdir -p /replicationsurveys
COPY . /replicationsurveys/
RUN ls -la /replicationsurveys/

# Install Java
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean;

# CMD ["sh", "-c", "cd ./replicationsurveys && python ./reproduce.py"]
COPY ./runreproduce.sh /runreproduce.sh
RUN chmod +x /runreproduce.sh

LABEL maintainer="Joe Bak-Coleman <jbakcoleman@gmail.com>"

RUN pip install -r /replicationsurveys/requirements.txt
RUN pip install nbconvert nbformat
CMD echo "Replication Crisis Model Container"


ENTRYPOINT ["/runreproduce.sh"]

# Default command (can be overridden)
CMD ["sh"]