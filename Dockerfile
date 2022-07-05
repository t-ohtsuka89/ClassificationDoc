FROM python:3.10-slim as compiler
ENV PYTHONUNBUFFERED=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN pip install -U tensorboard


FROM python:3.10-slim
COPY --from=compiler /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /logs
ENTRYPOINT ["tensorboard", "--logdir", "./", "--host", "0.0.0.0"]