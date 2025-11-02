import os
import logging
from dotenv import load_dotenv
load_dotenv() # Load environment variables from a .env file. This is crucial for keeping sensitive data like API keys out of your main codebase.

# Suppress most ADK internal logs to keep the console clean during Streamlit runs.
# You can change this to logging.INFO or logging.DEBUG for more verbose output during debugging.
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__) 

MODEL_GEMINI = "gemini-2.0-flash" # Specifies the Google Gemini model to be used by the ADK agent.
APP_NAME_FOR_ADK = "healthcare_test_agent" # A unique name for your application within ADK, used for session management.
USER_ID = "yogeshdhome" # A default user ID. In a real application, this would be dynamic (e.g., from a login system).

# Defines the initial state for new ADK sessions. This provides default values for user information.
INITIAL_STATE = {
    'context': None,
    'test_cases': [],
    'gap_analysis': "",
    'jira_csv': "",
    'processing_complete': False,
    'last_error': None,
}

MESSAGE_HISTORY_KEY = "messages_final_mem_v2" # Key used by Streamlit to store the chat history in its session state.
ADK_SESSION_KEY = "adk_session_id" # Key used by Streamlit to store the unique ADK session ID.

def get_api_key():
    """Retrieves the Google API Key from environment variables."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    # Basic check to ensure the key is present and not the placeholder.
    if not api_key or "YOUR_GOOGLE_API_KEY" in api_key or "your-google-api-key" in api_key.lower():
        return None
    return api_key

def configure_adk_authentication():
    """
    Configure ADK authentication based on available credentials.
    Sets environment variables for Google ADK to use.
    
    Priority:
    1. GOOGLE_API_KEY (for Google AI API) - simplest for local development
    2. Vertex AI (GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION)
    """
    api_key = get_api_key()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    
    # Check if using Vertex AI
    use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"
    
    if use_vertexai and project and location:
        # Use Vertex AI
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        logger.info(f"ADK configured to use Vertex AI: project={project}, location={location}")
        return {"method": "vertexai", "project": project, "location": location}
    elif api_key:
        # Use API key (simpler for local development)
        os.environ["GOOGLE_API_KEY"] = api_key
        logger.info("ADK configured to use Google AI API with API key")
        return {"method": "api_key", "api_key": api_key[:10] + "..."}  # Don't log full key
    else:
        logger.error("No authentication method configured. Set either GOOGLE_API_KEY or (GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION)")
        return None

# Configure authentication on import
_authentication_config = configure_adk_authentication()