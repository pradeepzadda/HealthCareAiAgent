"""
SQLite database module for storing session data.
This replaces Firestore for local development and simpler deployment.
"""
import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database file path
DB_FILE = os.path.join(os.path.dirname(__file__), 'healthcare_agent.db')


class Database:
    """SQLite database wrapper for session data storage."""
    
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Create test_sessions table with all columns
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS test_sessions (
                        session_id TEXT PRIMARY KEY,
                        context_data TEXT NOT NULL,
                        test_cases TEXT,
                        gap_analysis TEXT,
                        jira_csv TEXT,
                        recommended_test_cases TEXT,
                        included_recommendations TEXT,
                        edited_recommendations TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Add new columns if they don't exist (for existing databases)
                columns_to_add = [
                    ('test_cases', 'TEXT'),
                    ('gap_analysis', 'TEXT'),
                    ('jira_csv', 'TEXT'),
                    ('recommended_test_cases', 'TEXT'),
                    ('included_recommendations', 'TEXT'),
                    ('edited_recommendations', 'TEXT'),
                    ('edited_test_cases', 'TEXT'),
                    ('included_test_cases', 'TEXT'),
                    ('finalized_test_cases', 'TEXT'),
                    ('is_finalized', 'INTEGER')
                ]
                
                for column_name, column_type in columns_to_add:
                    try:
                        cursor.execute(f'''
                            ALTER TABLE test_sessions 
                            ADD COLUMN {column_name} {column_type}
                        ''')
                        logger.info(f"Added column {column_name} to test_sessions table")
                    except sqlite3.OperationalError as e:
                        # Column already exists, which is fine
                        if 'duplicate column name' not in str(e).lower():
                            logger.warning(f"Could not add column {column_name}: {e}")
                
                # Create index for faster lookups
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON test_sessions(session_id)
                ''')
                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_file}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_session_data(self, session_id: str, data: Dict[Any, Any]) -> bool:
        """
        Store session data in the database.
        
        Args:
            session_id: Unique session identifier
            data: Dictionary of data to store (can contain context_data, test_cases, gap_analysis, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract context_data (requirements and context) - keep existing structure
            context_data = {
                'requirements': data.get('requirements', []),
                'context': data.get('context', ''),
                'requirement_contexts': data.get('requirement_contexts', {})
            }
            context_json = json.dumps(context_data)
            
            # Extract other data fields
            test_cases_json = json.dumps(data.get('test_cases', [])) if data.get('test_cases') else None
            gap_analysis_text = data.get('gap_analysis', '')
            jira_csv_text = data.get('jira_csv', '')
            recommended_test_cases_json = json.dumps(data.get('recommended_test_cases', [])) if data.get('recommended_test_cases') else None
            included_recommendations_json = json.dumps(data.get('included_recommendations', {})) if data.get('included_recommendations') else None
            edited_recommendations_json = json.dumps(data.get('edited_recommendations', {})) if data.get('edited_recommendations') else None
            edited_test_cases_json = json.dumps(data.get('edited_test_cases', {})) if data.get('edited_test_cases') else None
            included_test_cases_json = json.dumps(data.get('included_test_cases', {})) if data.get('included_test_cases') else None
            finalized_test_cases_json = json.dumps(data.get('finalized_test_cases', [])) if data.get('finalized_test_cases') else None
            is_finalized = 1 if data.get('is_finalized', False) else 0
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Use INSERT OR REPLACE to handle both new and existing sessions
                # Update all columns, preserving context_data structure
                cursor.execute('''
                    INSERT OR REPLACE INTO test_sessions 
                    (session_id, context_data, test_cases, gap_analysis, jira_csv, 
                     recommended_test_cases, included_recommendations, edited_recommendations,
                     edited_test_cases, included_test_cases, finalized_test_cases, is_finalized, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (session_id, context_json, test_cases_json, gap_analysis_text, jira_csv_text,
                      recommended_test_cases_json, included_recommendations_json, edited_recommendations_json,
                      edited_test_cases_json, included_test_cases_json, finalized_test_cases_json, is_finalized))
                conn.commit()
                logger.info(f"Session data stored for session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to store session data: {e}")
            return False
    
    def store_complete_session(self, session_id: str, context_data: Dict[Any, Any], 
                               test_cases: list = None, gap_analysis: str = None,
                               jira_csv: str = None, recommended_test_cases: list = None,
                               included_recommendations: dict = None,
                               edited_recommendations: dict = None) -> bool:
        """
        Store complete session data with all fields.
        
        Args:
            session_id: Unique session identifier
            context_data: Requirements and context dictionary
            test_cases: List of generated test cases
            gap_analysis: Gap analysis markdown text
            jira_csv: Jira formatted CSV string
            recommended_test_cases: List of recommended test cases from gap analysis
            included_recommendations: Dictionary mapping recommendation indices to include/exclude state
            edited_recommendations: Dictionary mapping recommendation indices to edited test case data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            context_json = json.dumps(context_data)
            test_cases_json = json.dumps(test_cases) if test_cases else None
            gap_analysis_text = gap_analysis or ''
            jira_csv_text = jira_csv or ''
            recommended_test_cases_json = json.dumps(recommended_test_cases) if recommended_test_cases else None
            included_recommendations_json = json.dumps(included_recommendations) if included_recommendations else None
            edited_recommendations_json = json.dumps(edited_recommendations) if edited_recommendations else None
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO test_sessions 
                    (session_id, context_data, test_cases, gap_analysis, jira_csv, 
                     recommended_test_cases, included_recommendations, edited_recommendations, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (session_id, context_json, test_cases_json, gap_analysis_text, jira_csv_text,
                      recommended_test_cases_json, included_recommendations_json, edited_recommendations_json))
                conn.commit()
                logger.info(f"Complete session data stored for session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to store complete session data: {e}")
            return False
    
    def get_session_data(self, session_id: str) -> Optional[Dict[Any, Any]]:
        """
        Retrieve session data from the database.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary of session data or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT context_data, test_cases, gap_analysis, jira_csv, 
                           recommended_test_cases, included_recommendations, edited_recommendations,
                           edited_test_cases, included_test_cases, finalized_test_cases, is_finalized
                    FROM test_sessions
                    WHERE session_id = ?
                ''', (session_id,))
                row = cursor.fetchone()
                if row:
                    result = json.loads(row['context_data'])  # Contains requirements and context
                    
                    # Add other fields if they exist
                    if row['test_cases']:
                        result['test_cases'] = json.loads(row['test_cases'])
                    if row['gap_analysis']:
                        result['gap_analysis'] = row['gap_analysis']
                    if row['jira_csv']:
                        result['jira_csv'] = row['jira_csv']
                    if row['recommended_test_cases']:
                        result['recommended_test_cases'] = json.loads(row['recommended_test_cases'])
                    if row['included_recommendations']:
                        result['included_recommendations'] = json.loads(row['included_recommendations'])
                    if row['edited_recommendations']:
                        result['edited_recommendations'] = json.loads(row['edited_recommendations'])
                    if row.get('edited_test_cases'):
                        result['edited_test_cases'] = json.loads(row['edited_test_cases'])
                    if row.get('included_test_cases'):
                        result['included_test_cases'] = json.loads(row['included_test_cases'])
                    if row.get('finalized_test_cases'):
                        result['finalized_test_cases'] = json.loads(row['finalized_test_cases'])
                    if row.get('is_finalized') is not None:
                        result['is_finalized'] = bool(row['is_finalized'])
                    
                    return result
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve session data: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM test_sessions
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
                logger.info(f"Session deleted: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def list_sessions(self, limit: int = 100) -> list:
        """
        List recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session IDs sorted by most recent
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id FROM test_sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                ''', (limit,))
                return [row['session_id'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if database is available and working."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                return True
        except Exception as e:
            logger.error(f"Database availability check failed: {e}")
            return False


# Global database instance
_db_instance = None


def get_database() -> Optional[Database]:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        try:
            _db_instance = Database()
            if _db_instance.is_available():
                logger.info("SQLite database initialized successfully")
                return _db_instance
            else:
                logger.warning("SQLite database not available")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite database: {e}")
            return None
    return _db_instance

