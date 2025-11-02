"""Conversation manager for multi-turn Q&A with data."""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
	"""Single turn in a conversation."""
	question: str
	sql: str
	headers: List[str]
	rows: List[Tuple]
	summary: str
	error: Optional[str] = None


class ConversationManager:
	"""Manages conversation history for iterative query refinement."""
	
	def __init__(self, max_history: int = 5):
		self.max_history = max_history
		self.history: List[ConversationTurn] = []
		self.csv_path: Optional[str] = None
		self.schema_prompt: Optional[str] = None
	
	def set_context(self, csv_path: str, schema_prompt: str):
		"""Set the current CSV context."""
		if csv_path != self.csv_path:
			# New CSV, clear history
			self.clear()
		self.csv_path = csv_path
		self.schema_prompt = schema_prompt
	
	def add_turn(self, turn: ConversationTurn):
		"""Add a conversation turn."""
		self.history.append(turn)
		# Keep only last N turns
		if len(self.history) > self.max_history:
			self.history = self.history[-self.max_history:]
	
	def get_last_turn(self) -> Optional[ConversationTurn]:
		"""Get the most recent turn."""
		return self.history[-1] if self.history else None
	
	def get_context_for_refinement(self, new_question: str) -> str:
		"""Build context string for query refinement."""
		if not self.history:
			return new_question
		
		# Check if this is a refinement request
		refinement_indicators = [
			'also', 'too', 'additionally', 'include', 'add', 'show',
			'missing', 'forgot', 'need', 'want', 'correct', 'fix',
			'change', 'update', 'modify', 'instead', 'rather',
			'group by', 'order by', 'sort', 'filter', 'where'
		]
		
		is_refinement = any(indicator in new_question.lower() for indicator in refinement_indicators)
		
		if not is_refinement:
			# New independent question
			return new_question
		
		# Build context from last turn
		last_turn = self.get_last_turn()
		context_parts = [
			"PREVIOUS QUERY CONTEXT:",
			f"Question: {last_turn.question}",
			f"SQL: {last_turn.sql}",
			f"Columns in SELECT: {', '.join(last_turn.headers)}",
			f"GROUP BY: {self._extract_group_by(last_turn.sql)}",
			"",
			"USER FEEDBACK / REFINEMENT REQUEST:",
			new_question,
			"",
			"**CRITICAL REFINEMENT RULES:**",
			"",
			"IF user asks to 'make it in crores/millions/thousands' or change units:",
			"  → ONLY modify the calculation (divide/multiply) and alias",
			"  → DO NOT add Year, Date, Category, or ANY other columns",
			"  → DO NOT change GROUP BY or column list",
			"  → Example: If previous SELECT had 'Category', keep ONLY 'Category', just change the math",
			"",
			"IF user asks to 'add <column_name>' or 'include <column_name>' or 'show <column_name>':",
			"  → ONLY then add that specific column",
			"  → Add it to SELECT and GROUP BY (if aggregating)",
			"",
			"IF user asks to 'remove <column_name>':",
			"  → Remove that column from SELECT and GROUP BY",
			"",
			"DEFAULT RULE: If unclear, keep the EXACT same columns as before. Only change calculations/filters/sorting.",
			"",
			"Generate an UPDATED SQL query following these rules."
		]
		
		return "\n".join(context_parts)
	
	def _extract_group_by(self, sql: str) -> str:
		"""Extract GROUP BY clause from SQL."""
		import re
		match = re.search(r'GROUP\s+BY\s+([^;]+?)(?:ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
		if match:
			return match.group(1).strip()
		return "None"
	
	def format_history_for_display(self) -> str:
		"""Format conversation history for display."""
		if not self.history:
			return "No previous queries in this session."
		
		lines = ["📝 Conversation History:", ""]
		for i, turn in enumerate(self.history, 1):
			lines.append(f"{i}. Q: {turn.question}")
			lines.append(f"   SQL: {turn.sql[:80]}..." if len(turn.sql) > 80 else f"   SQL: {turn.sql}")
			lines.append(f"   Result: {len(turn.rows)} row(s)")
			if turn.error:
				lines.append(f"   ❌ Error: {turn.error}")
			lines.append("")
		
		return "\n".join(lines)
	
	def clear(self):
		"""Clear conversation history."""
		self.history = []
		self.csv_path = None
		self.schema_prompt = None
	
	def is_refinement_request(self, question: str) -> bool:
		"""Detect if the question is a refinement of the previous query."""
		if not self.history:
			return False
		
		refinement_patterns = [
			r'\b(also|too|additionally)\b',
			r'\b(include|add|show)\s+(the\s+)?(\w+\s+)?(name|column|field)',
			r'\bmissing\b',
			r'\b(correct|fix|update|modify|change)\b',
			r'\binstead\b',
			r'\bgroup\s+by\b',
			r'\border\s+by\b',
			r'\bsort\s+by\b',
			r'\bfilter\b',
			r'\bwhere\b'
		]
		
		import re
		return any(re.search(pattern, question.lower()) for pattern in refinement_patterns)
	
	def validate_refinement(self, new_sql: str, request: str) -> Tuple[bool, Optional[str]]:
		"""
		Validate that the refined SQL actually followed the user's request.
		Returns (is_valid, error_message).
		"""
		if not self.history:
			return True, None
		
		last_turn = self.get_last_turn()
		request_lower = request.lower()
		
		# Extract columns from both queries
		import re
		
		def extract_select_columns(sql):
			# Simple extraction - get column names from SELECT clause
			match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
			if match:
				select_clause = match.group(1)
				# Get column names (look for AS aliases or bare column names)
				cols = []
				for part in select_clause.split(','):
					# Look for AS alias
					as_match = re.search(r'AS\s+(\w+)', part, re.IGNORECASE)
					if as_match:
						cols.append(as_match.group(1))
					else:
						# Get last word that looks like a column
						words = part.strip().split()
						if words:
							cols.append(words[-1])
				return cols
			return []
		
		old_cols = set(extract_select_columns(last_turn.sql))
		new_cols = set(extract_select_columns(new_sql))
		
		# Check for unit change requests
		unit_keywords = ['crore', 'million', 'thousand', 'lakh', 'billion', 'percent']
		is_unit_change = any(keyword in request_lower for keyword in unit_keywords)
		
		if is_unit_change:
			# For unit changes, columns should stay the same
			added_cols = new_cols - old_cols
			if added_cols:
				# Check if added columns were explicitly requested
				add_keywords = ['add', 'include', 'show', 'also']
				if not any(keyword in request_lower for keyword in add_keywords):
					return False, f"Added unrequested columns: {', '.join(added_cols)}. User only asked to change units."
		
		return True, None
	
	def get_hints_for_refinement(self, question: str) -> List[str]:
		"""Generate hints for common refinement requests."""
		hints = []
		question_lower = question.lower()
		
		last_turn = self.get_last_turn()
		if not last_turn:
			return hints
		
		# Check for unit change (should NOT add columns)
		unit_keywords = ['crore', 'million', 'thousand', 'lakh', 'billion', 'percent']
		if any(keyword in question_lower for keyword in unit_keywords):
			hints.append("💡 Detected unit change request - will ONLY modify calculations, not add columns")
		
		# Check for missing column requests
		if any(word in question_lower for word in ['name', 'category', 'type', 'group', 'add', 'include', 'show']):
			if not any(unit in question_lower for unit in unit_keywords):
				hints.append("💡 Tip: Will add the requested column to results")
		
		# Check for sorting requests
		if any(word in question_lower for word in ['sort', 'order', 'top', 'bottom']):
			hints.append("💡 Tip: Will adjust ORDER BY clause")
		
		# Check for filtering requests
		if any(word in question_lower for word in ['only', 'where', 'filter', 'exclude']):
			hints.append("💡 Tip: Will add WHERE clause to filter data")
		
		# Check for limit requests
		if any(word in question_lower for word in ['top', 'first', 'last', 'limit']):
			hints.append("💡 Tip: Will add LIMIT clause")
		
		return hints

