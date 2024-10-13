FROM python:3.10-slim-bookworm

WORKDIR /home/app

COPY requirements.txt src /home/app/

RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 app  \
    && useradd --uid 10001 --gid app app \
    && chown -R app:app /home/app

EXPOSE 8000

USER app

CMD ["fastapi", "run"]
