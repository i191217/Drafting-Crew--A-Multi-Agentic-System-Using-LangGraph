FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI server (adjust if your entry point is different)
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]