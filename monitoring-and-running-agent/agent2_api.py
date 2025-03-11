"""FastAPI-based Coding Agent Module."""
import subprocess
import requests
import uvicorn
from fastapi import FastAPI

app = FastAPI()

AGENT_1_URL = "http://agent1_endpoint_url"  # Replace with the actual Agent 1 endpoint

class CodingAgent:
    """A coding agent that generates and runs code."""
    def __init__(self):
        self.default_responses = [
            "Let me think about that...",
            "That's an interesting request!",
            "I'll need a moment to process that."
        ]

    def generate_code(self, task_description):
        """Generates a Python script based on the task description."""
        if "hello world" in task_description.lower():
            return "print('Hello, World!')"
        if "add two numbers" in task_description.lower():
            return (
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n\n"
                "if __name__ == '__main__':\n"
                "    print(add(3, 5))"
            )
        return "# Unable to generate code for the given task."

    def run_code(self, code):
        """Runs the generated Python code and returns the output."""
        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as call_error:
            return f"Error: {call_error.stderr.strip()}"

    def fetch_database_info(self):
        """Fetches the information database from Agent 1."""
        try:
            response = requests.get(f"{AGENT_1_URL}/database", timeout=5)
            return response.json()
        except requests.exceptions.RequestException as request_error:
            return {"error": str(request_error)}

    def trigger_agent_1(self, data):
        """Sends a request to Agent 1 to continue processing."""
        try:
            response = requests.post(f"{AGENT_1_URL}/process", json={"data": data}, timeout=5)
            return response.json()
        except requests.exceptions.RequestException as request_error:
            return {"error": str(request_error)}

agent = CodingAgent()

@app.get("/health")
def health_check():
    """Endpoint to check system health."""
    return {"status": "ok", "message": "Coding Agent 2 is running."}

@app.post("/generate")
def generate_code(task_description: str):
    """Endpoint to generate code from a task description."""
    return {"code": agent.generate_code(task_description)}

@app.post("/run")
def run_code(code: str):
    """Endpoint to run the generated code."""
    return {"output": agent.run_code(code)}

@app.post("/trigger_agent_1")
def trigger_agent_1(data: str):
    """Endpoint to send data to Agent 1 for further processing."""
    response = agent.trigger_agent_1(data)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

# Unit Tests

def test_generate_code():
    """Tests the code generation function."""
    assert agent.generate_code("hello world") == "print('Hello, World!')"
    assert "def add(a: int, b: int)" in agent.generate_code("add two numbers")
    assert "# Unable to generate code" in agent.generate_code("unknown task")

def test_run_code():
    """Tests the code execution function."""
    assert agent.run_code("print('Hello, World!')") == "Hello, World!"
    assert agent.run_code("print(2 + 2)") == "4"
    assert "Error" in agent.run_code("invalid_code")

def test_fetch_database_info():
    """Tests fetching database info from Agent 1."""
    response = agent.fetch_database_info()
    assert isinstance(response, dict)


# things we might need:
"""
way to test evaluation
hyperparameters
"""