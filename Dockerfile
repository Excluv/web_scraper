# Python 
FROM python:3.12
WORKDIR /scraper

COPY requirements.txt requirements.txt
RUN pip install playwright
RUN playwright install chromium
