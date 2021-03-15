FROM python:3.8.5-slim as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH .
ENV PIPENV_VENV_IN_PROJECT=1

FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv
RUN apt-get update && apt-get install -y gcc

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install

FROM base AS runtime
# Copy virtual env from python-deps stage
COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN useradd --create-home modelmanagement
WORKDIR /home/modelmanagement
USER modelmanagement

# Install application into container
COPY . .

# Run the application
CMD ["python", "app/main.py"]