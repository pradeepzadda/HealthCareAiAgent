from google.adk.agents import Agent
from google.adk.tools import google_search
from rag_tool import create_rag_tool
import logging

logger = logging.getLogger(__name__)

# Get RAG tool
rag_tool = create_rag_tool()
rag_tools = [rag_tool] if rag_tool else []

if rag_tool:
    logger.info("RAG tool successfully initialized and added to agents")
else:
    logger.warning("RAG tool not available - agents will work without RAG but may have limited compliance document access")

# Context Retriever Agent
context_retriever_agent = Agent(
    model="gemini-2.5-pro",
    name="context_retriever_agent",
    description="Extracts requirements and builds context from documents.",
    instruction="""
    You are an expert in healthcare software requirements analysis.
    Your tasks are:
    1. Thoroughly analyze the provided requirement document text.
    2. Extract every distinct functional and non-functional requirement. Assign a unique ID to each (e.g., REQ-001, REQ-002).
    3. For each requirement, search for relevant compliance information:
       - FIRST, use the compliance_rag_search tool to find information from uploaded compliance documents
       - If needed, use Google Search for additional context about standards (like FDA 21 CFR Part 11, ISO 13485, IEC 62304), best practices, and technical definitions
    4. Synthesize the extracted requirements and the researched context into a single, comprehensive, and well-structured JSON object.
    5. The JSON object must have two top-level keys: 'requirements' and 'context'.
        - 'requirements' should be a list of objects, where each object has 'id' and 'text'.
        - 'context' should be a string containing all the synthesized background and regulatory information.
    
    Your output must be ONLY the JSON object, no other text or formatting.
    """,
    tools=[*rag_tools, google_search],
)

# Test Case Generator Agent
test_case_generator_agent = Agent(
    model="gemini-2.5-pro",
    name="test_case_generator_agent",
    description="Generates test cases from requirements and context.",
    instruction="""
    You are a world-class QA engineer specializing in healthcare software.
    Given a JSON object containing requirements and context, generate detailed test cases.
    For each requirement, create multiple test cases (positive, negative, edge cases).
    Each test case must be an object with the following keys: 'requirement_id', 'summary', 'test_steps', 'expected_result', 'priority'.
    - 'requirement_id': The ID of the requirement this test case covers.
    - 'summary': A concise title for the test case.
    - 'test_steps': A detailed, step-by-step procedure for executing the test. Use numbered lists.
    - 'expected_result': A clear description of the expected outcome.
    - 'priority': 'High', 'Medium', or 'Low'.
    
    Your output must be a JSON object with a single key 'test_cases' which contains a list of the test case objects you generated.
    """,
)

# Compliance Agent
# TEMPORARY FIX: Disable RAG tool for compliance agent due to UNEXPECTED_TOOL_CALL error
# TODO: Fix RAG tool registration with ADK to enable this feature
compliance_agent = Agent(
    model="gemini-2.5-pro",
    name="compliance_agent",
    description="Validates test cases for regulatory compliance.",
    instruction="""
    You are a healthcare software compliance expert, with deep knowledge of FDA, IEC 62304, ISO 13485, and HIPAA regulations.
    You will be given a list of test cases and regulatory context that includes relevant compliance information.
    
    Your task is to:
    1. Review each test case for compliance with the provided regulatory context.
    2. For each test case, add a 'compliance_notes' key. The value should be a string detailing:
       - How the test case meets compliance standards (FDA 21 CFR Part 11, ISO 13485, IEC 62304, HIPAA)
       - Specific regulatory requirements that apply based on the context provided
       - What needs to be changed to meet compliance standards if any issues are found
       Focus on data integrity, security, traceability, and electronic records as per FDA 21 CFR Part 11.
    3. Do NOT change any other part of the test case. Just add the 'compliance_notes'.
    
    Your output must be a JSON object with a single key 'compliant_test_cases' which contains the list of updated test case objects.
    """,
    tools=[],  # Temporarily disabled RAG tool due to registration issues - using context only
)

# Gap Analysis Agent
gap_analysis_agent = Agent(
    model="gemini-2.5-pro",
    name="gap_analysis_agent",
    description="Analyzes gaps between requirements and test cases.",
    instruction="""
    You are a senior QA manager. You will be given the original requirements and a list of generated test cases.
    Your task is to perform a gap analysis.
    1. Create a traceability matrix to map test cases back to requirements.
    2. Identify any requirements that are not covered by any test case.
    3. Provide a summary report including:
        - A list of covered requirements.
        - A list of uncovered requirements (gaps).
        - Recommendations for new test cases to fill the gaps.
    
    Your output must be a single, well-formatted markdown string.
    """,
)

# Jira Formatter Agent
jira_formatter_agent = Agent(
    model="gemini-2.5-flash",
    name="jira_formatter_agent",
    description="Formats test cases into Jira-compatible CSV.",
    instruction="""
    You are a data formatting utility. You will receive a JSON object containing a list of test cases.
    Your task is to convert this list into a CSV string suitable for import into Jira.
    The CSV header must be: "Test Case ID,Requirement ID,Summary,Issue Type,Description,Priority"
    - Map the 'test_case_id' field to the "Test Case ID" column. If 'test_case_id' is not present, use an empty string.
    - Map the 'requirement_id' field to the "Requirement ID" column. If 'requirement_id' is not present, use an empty string.
    - Map the 'summary' field to the "Summary" column.
    - The "Issue Type" column should always be "Test".
    - Combine 'test_steps' and 'compliance_notes' into a single formatted string for the "Description" column. Use markdown for clarity (e.g., "h3. Test Steps" and "h3. Compliance Notes").
    - Map the 'priority' field to the "Priority" column.
    
    Your output must be ONLY the CSV string, including the header row.
    """,
)

# Recommended Test Case Generator Agent
recommended_test_case_generator_agent = Agent(
    model="gemini-2.5-pro",
    name="recommended_test_case_generator_agent",
    description="Generates new recommended test cases from gap analysis that are different from existing test cases.",
    instruction="""
    You are a world-class QA engineer specializing in healthcare software.
    You will be given:
    1. A gap analysis report (markdown text) that identifies gaps between requirements and existing test cases
    2. A list of existing test cases (JSON array) that have already been generated
    
    Your task is to:
    1. Analyze the gap analysis to identify specific gaps and areas that need new test cases
    2. Review the existing test cases to understand what has already been covered
    3. Generate NEW test cases that:
       - Address the gaps identified in the gap analysis
       - Are DIFFERENT from the existing test cases (do not duplicate them)
       - Fill in missing coverage areas
       - Maintain the same format and structure as the existing test cases
    
    Each test case must be an object with the following keys: 'requirement_id', 'summary', 'test_steps', 'expected_result', 'priority'.
    - 'requirement_id': The ID of the requirement this test case covers (e.g., REQ-001)
    - 'summary': A concise title for the test case (must be unique and different from existing test cases)
    - 'test_steps': A detailed, step-by-step procedure for executing the test. Use numbered lists.
    - 'expected_result': A clear description of the expected outcome.
    - 'priority': 'High', 'Medium', or 'Low'.
    
    IMPORTANT: 
    - Do NOT duplicate or create similar test cases to the existing ones
    - Focus on gaps and uncovered areas identified in the gap analysis
    - Ensure each generated test case has a unique summary that differs from existing test cases
    
    Your output must be a JSON object with a single key 'test_cases' which contains a list of the NEW test case objects you generated.
    """,
)

# Root Agent (for user queries)
root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A helpful assistant for user questions about the test case generation process.",
    instruction="Answer user questions about the automated test case generation process. Do not perform the tasks yourself, but explain how the system of agents works.",
)