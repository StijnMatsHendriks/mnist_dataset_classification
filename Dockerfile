FROM nvcr.io/nvidia/pytorch:20.12-py3
WORKDIR /app
COPY requirements.txt gpu_available.py /app/
EXPOSE 8888
RUN pip install -r requirements.txt
CMD ["python", "gpu_available.py"]
CMD ["python", "ml_perceptron.py"]
