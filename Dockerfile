FROM python:3.10
WORKDIR /app
COPY . .
RUN apt update && apt -y install openjdk-17-jdk
ENV POETRY_HOME=/opt/poetry
ENV PATH="$PATH:$POETRY_HOME/bin"
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0
RUN poetry update --with dev,test
RUN poetry run pre-commit install
RUN curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
