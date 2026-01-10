"""Prompts for extraction agent."""
EXTRACTION_PROMPT = """You are an extractor agent.
Generate {max_queries} queries to retrieve evidence about the character.

Book: {book_name}
Character: {character_name}
Backstory: {backstory}

Return only a JSON list of strings, each string being a query.
Example: ["query 1", "query 2", "query 3"]
"""
