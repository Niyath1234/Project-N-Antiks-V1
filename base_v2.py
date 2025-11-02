#!/usr/bin/env python3
"""
Simplified Multi-Agent Text-to-SQL Analytics CLI
Clean, efficient, and working.
"""
import os
import sys
import ollama

from sql_runner import create_connection, register_numeric_udfs, format_table, execute_query, is_safe_sql
from data_loader import load_csv_to_sqlite, generate_schema_prompt
from conversation_manager import ConversationManager, ConversationTurn
from custom_rules import get_rule_manager

# Configuration
host = os.environ.get('OLLAMA_HOST')
model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
csv_path = os.environ.get('CSV_PATH')

# Ollama client
if host:
	client = ollama.Client(host=host)
	chat_fn = client.chat
else:
	chat_fn = ollama.chat


def extract_sql(text: str) -> str:
	"""Extract SQL from model output."""
	import re
	m = re.search(r"```sql\n([\s\S]*?)```", text, flags=re.IGNORECASE)
	if m:
		return m.group(1).strip()
	m2 = re.search(r"```\n([\s\S]*?)```", text)
	if m2:
		return m2.group(1).strip()
	return text.strip()


def run_query(conn, system_prompt: str, question: str, conversation: ConversationManager):
	"""Run a single query through the system."""
	
	# Check for refinement
	is_refinement = conversation.is_refinement_request(question)
	
	if is_refinement:
		enhanced_question = conversation.get_context_for_refinement(question)
	else:
		enhanced_question = question
	
	# Build messages
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": enhanced_question}
	]
	
	# Get SQL from LLM
	try:
		response = chat_fn(model=model, messages=messages)
		content = response['message']['content'].strip()
	except Exception as e:
		print(f"\n❌ Error connecting to Ollama: {e}")
		print("💡 Make sure Ollama is running: ollama serve")
		return
	
	# Extract SQL
	sql = extract_sql(content)
	
	# Safety check
	if not is_safe_sql(sql):
		print(f"\n❌ SQL failed safety check (must be single read-only SELECT/WITH)")
		return
	
	# Execute
	try:
		headers, rows = execute_query(conn, sql)
	except Exception as e:
		print(f"\n❌ SQL execution error: {e}")
		print(f"\nGenerated SQL:\n{sql}")
		return
	
	# Success!
	print(f"\n✅ Query executed successfully!")
	print(f"\n📊 SQL Query:")
	print(f"```sql\n{sql}\n```")
	print(f"\n📋 Results:")
	print(format_table(headers, rows))
	
	# Generate summary
	try:
		summary_msgs = [
			{"role": "system", "content": "You are a data analyst. Provide a concise 1-2 sentence summary."},
			{"role": "user", "content": (
				f"Question: {question}\n"
				f"SQL: {sql}\n"
				f"Results:\n{','.join(headers)}\n" + 
				"\n".join(",".join(str(c) for c in r) for r in rows[:50])
			)}
		]
		summary_resp = chat_fn(model=model, messages=summary_msgs)
		summary = summary_resp['message']['content'].strip()
		print(f"\n💬 Summary:")
		print(summary)
	except:
		summary = ""
	
	# Store in conversation
	turn = ConversationTurn(
		question=question,
		sql=sql,
		headers=headers,
		rows=rows,
		summary=summary,
		error=None
	)
	conversation.add_turn(turn)


try:
	# Initialize
	conn = create_connection(":memory:")
	register_numeric_udfs(conn)
	
	# Load CSV
	if not csv_path:
		print("❌ No CSV provided. Set CSV_PATH environment variable.")
		sys.exit(1)
	
	if not os.path.exists(csv_path):
		print(f"❌ CSV not found: {csv_path}")
		sys.exit(1)
	
	table_name, cols = load_csv_to_sqlite(conn, csv_path, table_name='dataset')
	system_prompt = generate_schema_prompt(table_name, cols)
	
	print(f"✅ Loaded CSV into table '{table_name}'")
	print(f"🤖 Simplified Analytics System Ready")
	print(f"   Model: {model}")
	print()
	print("Commands:")
	print("  :load <path>       - Load new CSV")
	print("  :history           - Show conversation")
	print("  :clear             - Clear history")
	print("  :rules             - Show custom rules")
	print("  :learn <rule>      - Add a learning rule (persists!)")
	print("  exit/quit          - Exit")
	print()
	print("💡 Tip: When the model makes a mistake, use ':learn <correction>' to teach it!")
	print()
	
	# Conversation manager
	conversation = ConversationManager(max_history=10)
	conversation.set_context(csv_path, system_prompt)
	
	# REPL loop
	while True:
		q = input('\nQ: ').strip()
		
		if not q:
			continue
		
		if q.lower() in {'exit', 'quit'}:
			break
		
		if q.lower().startswith(':load '):
			new_csv = q[6:].strip()
			if not os.path.exists(new_csv):
				print(f"❌ File not found: {new_csv}")
				continue
			try:
				conn.close()
				conn = create_connection(":memory:")
				register_numeric_udfs(conn)
				table_name, cols = load_csv_to_sqlite(conn, new_csv, table_name='dataset')
				system_prompt = generate_schema_prompt(table_name, cols)
				conversation = ConversationManager(max_history=10)
				conversation.set_context(new_csv, system_prompt)
				print(f"✅ Loaded new CSV: {new_csv}")
			except Exception as e:
				print(f"❌ Failed to load CSV: {e}")
			continue
		
		if q.lower() == ':clear':
			conversation = ConversationManager(max_history=10)
			conversation.set_context(csv_path, system_prompt)
			print("✅ Conversation history cleared")
			continue
		
		if q.lower() == ':history':
			if conversation.history:
				print("\n📝 Conversation History:\n")
				for i, turn in enumerate(conversation.history, 1):
					print(f"{i}. Q: {turn.question}")
					print(f"   SQL: {turn.sql[:80]}...")
					print(f"   Columns: {', '.join(turn.headers)}")
					print()
			else:
				print("No conversation history")
			continue
		
		if q.lower() == ':rules':
			rm = get_rule_manager()
			print(rm.format_rules_for_display())
			continue
		
		if q.lower().startswith(':learn '):
			# Add a learning rule from user feedback
			rule_text = q[7:].strip()
			if rule_text:
				rm = get_rule_manager()
				rule_id = rm.add_rule(rule_text, category='learned')
				print(f"✅ Learned new rule #{rule_id}: {rule_text}")
				print("💡 This rule will now guide future queries")
				# Regenerate prompt with new rule
				system_prompt = generate_schema_prompt(table_name, cols)
			else:
				print("❌ Please provide a rule to learn")
				print("Example: :learn Always use CLEAN_NUMERIC for numeric columns")
			continue
		
		# Process query
		run_query(conn, system_prompt, q, conversation)

except KeyboardInterrupt:
	print("\n\nGoodbye!")
except Exception as e:
	print(f"\n❌ Error: {e}")
	print(f"\nHint: Ensure Ollama is running (ollama serve) and model '{model}' is pulled.")
