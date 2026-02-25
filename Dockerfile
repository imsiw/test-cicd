FROM python:3.14-slim
WORKDIR /app
COPY . .
RUN pip install gradio
EXPOSE 7860
CMD python app.py