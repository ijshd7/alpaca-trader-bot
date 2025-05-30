FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PAPER_API_BASE_URL=${PAPER_API_BASE_URL}
ENV PAPER_API_KEY=${PAPER_API_KEY}
ENV PAPER_API_SECRET=${PAPER_API_SECRET}

CMD ["python", "-u", "trader_bot.py"]