# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories
COPY src/ /app/src/
COPY app/ /app/app/
COPY models/ /app/models/
COPY data/ /app/data/

# Expose port 8000 for the FastAPI service
EXPOSE 8000

# Set environment variable to ensure immediate stdout logging
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
