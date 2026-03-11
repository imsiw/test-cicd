FROM python:3.14-slim
WORKDIR /app
COPY . .
RUN pip install gradio
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
EXPOSE 7860
CMD python app.py
