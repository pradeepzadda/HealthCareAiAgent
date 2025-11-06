# Auto Test Agent

The Auto Test Agent is an AI-powered application built with the Google Agent Development Kit (ADK). It provides a user-friendly Streamlit interface to interact with an agent capable of processing documents (`.pdf`, `.docx`) and data files (`.csv` via pandas) to assist with testing-related tasks.

## Features

- Interactive web UI powered by Streamlit.
- Core agent logic built with `google-adk`.
- Utilizes Google's Generative AI models (`google-generativeai`).
- Document processing for PDF (`pypdf`) and Word (`python-docx`).
- Data analysis capabilities with `pandas`.
- Local SQLite database for session data persistence..

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11+
- `pip` (Python package installer)
- Google Cloud SDK

You will also need a Google Cloud project with Vertex AI API enabled (for ADK and Gemini models).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Auto_Test_Agent
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application uses a `.env` file to manage environment variables.

1.  Create a `.env` file in the root of the project directory:
    ```bash
    touch .env
    ```

2.  Add your Google Cloud configuration to the `.env` file. This is crucial for the ADK and Google client libraries to authenticate and connect to the correct project.
    ```env
    GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    GOOGLE_CLOUD_LOCATION="your-gcp-region"
    GOOGLE_API_KEY="your-google-api-key"
    ```

**Note**: The application uses SQLite for local session data storage. A database file (`healthcare_agent.db`) will be automatically created in the project directory on first run. No additional database setup is required.

## Running the Application

To run the Streamlit web application locally, use the following command. Replace `app.py` with the name of your main Streamlit script.

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Deployment.

You can deploy this agent to Google Cloud using the ADK CLI. The following command deploys the agent as a web app to Cloud Run.

```bash
adk deploy cloud-run --agent_folder=. --with_ui
```
