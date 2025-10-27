FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the rest of the code
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860

# Run the Gradio app (adjust if your entry point is different)
CMD ["python", "app.py"]