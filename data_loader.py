import csv
import os
import sqlite3
from typing import List, Tuple
from custom_rules import get_rule_manager


NUMERIC_SYMBOLS = set([',', '$', '₹', '£', '€', '%'])


def _infer_type(values: List[str]) -> str:
	"""Infer SQLite type from sample string values."""
	is_int = True
	is_float = True
	for v in values:
		if v is None or v == '':
			continue
		s = v.strip()
		# remove common numeric symbols for inference
		for sym in NUMERIC_SYMBOLS:
			s = s.replace(sym, '')
		try:
			int(s)
		except Exception:
			is_int = False
			try:
				float(s)
			except Exception:
				is_float = False
		if not is_int and not is_float:
			return 'TEXT'
	if is_int:
		return 'INTEGER'
	if is_float:
		return 'REAL'
	return 'TEXT'


def _sanitize_cell_for_numeric(cell: str) -> str:
	if cell is None:
		return None
	s = cell.strip()
	if s == '':
		return None
	# remove thousands separators and currency/percent signs
	for sym in NUMERIC_SYMBOLS:
		s = s.replace(sym, '')
	return s if s != '' else None


def _normalize_rows(rows: List[List[str]], numeric_mask: List[bool]) -> List[List[object]]:
	"""Convert numeric-looking cells to clean strings (or None) for SQLite to coerce."""
	normalized = []
	for row in rows:
		new_row: List[object] = []
		for i, cell in enumerate(row):
			if numeric_mask[i]:
				new_row.append(_sanitize_cell_for_numeric(cell))
			else:
				new_row.append(cell if cell != '' else None)
		normalized.append(new_row)
	return normalized


def load_csv_to_sqlite(conn: sqlite3.Connection, csv_path: str, table_name: str = 'csv_table', sample_rows: int = 1000) -> Tuple[str, List[str]]:
	"""Create a table and load the CSV into SQLite. Returns (table_name, columns)."""
	if not os.path.exists(csv_path):
		raise FileNotFoundError(csv_path)
	with open(csv_path, 'r', encoding='utf-8', newline='') as f:
		reader = csv.reader(f)
		headers = next(reader)
		samples: List[List[str]] = [[] for _ in headers]
		buffer_rows: List[List[str]] = []
		for i, row in enumerate(reader):
			buffer_rows.append(row)
			for idx, cell in enumerate(row):
				samples[idx].append(cell)
			if i + 1 >= sample_rows:
				break
		# Infer types
		types = [_infer_type(col_vals) for col_vals in samples]
		numeric_mask = [t in ('INTEGER', 'REAL') for t in types]
		# Create table
		quoted_cols = [f'"{col}" {typ}' for col, typ in zip(headers, types)]
		schema_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (' + ', '.join(quoted_cols) + ');'
		conn.execute(schema_sql)
		conn.commit()
		# Load data (including already buffered rows and the rest)
		placeholders = ','.join(['?'] * len(headers))
		insert_sql = f'INSERT INTO "{table_name}" (' + ','.join([f'"{h}"' for h in headers]) + f') VALUES ({placeholders})'
		# Insert buffered (normalized)
		if buffer_rows:
			norm_buffer = _normalize_rows(buffer_rows, numeric_mask)
			conn.executemany(insert_sql, norm_buffer)
		# Insert remaining
		with open(csv_path, 'r', encoding='utf-8', newline='') as rf:
			rdr2 = csv.reader(rf)
			next(rdr2, None)  # skip header
			# skip already buffered rows
			for _ in range(len(buffer_rows)):
				try:
					next(rdr2)
				except StopIteration:
					break
			batch: List[List[str]] = []
			for row in rdr2:
				batch.append(row)
				if len(batch) >= 1000:
					norm_batch = _normalize_rows(batch, numeric_mask)
					conn.executemany(insert_sql, norm_batch)
					batch = []
			if batch:
				norm_batch = _normalize_rows(batch, numeric_mask)
				conn.executemany(insert_sql, norm_batch)
		conn.commit()
	return table_name, headers


def generate_schema_prompt(table_name: str, columns: List[str], include_custom_rules: bool = True) -> str:
	"""Generate schema prompt by loading base prompt file and adding table-specific schema + custom rules."""
	
	# Load base prompt from file
	prompt_file = os.path.join(os.path.dirname(__file__), "schema_prompt.txt")
	if os.path.exists(prompt_file):
		with open(prompt_file, 'r', encoding='utf-8') as f:
			base_prompt = f.read()
	else:
		# Fallback to basic prompt if file doesn't exist
		base_prompt = "You are a text-to-SQL generator for SQLite."
	
	# Heuristic: columns likely numeric by name
	suspect_numeric_tokens = ['amount', 'value', 'total', 'sum', 'revenue', 'sales', 'qty', 'quantity', 'price', 'cost', 'count', 'percent', 'rate', 'avg', 'median']
	suspect_numeric_cols = [c for c in columns if any(tok in c.lower() for tok in suspect_numeric_tokens)]
	
	# Build table-specific schema section
	schema_lines = [
		"",
		"="*80,
		"**YOUR CURRENT TABLE SCHEMA:**",
		f"Table name: {table_name}",
		"",
		"Columns:",
	]
	for c in columns:
		schema_lines.append(f"  - {c}")
	
	if suspect_numeric_cols:
		schema_lines.extend([
			"",
			"Suspected numeric columns (apply CLEAN_NUMERIC):",
			*[f"  - {c}" for c in suspect_numeric_cols]
		])
	
	schema_lines.append("="*80)
	
	# Combine base prompt + schema + custom rules
	final_prompt = base_prompt + "\n" + "\n".join(schema_lines)
	
	# Append custom rules if enabled
	if include_custom_rules:
		rule_manager = get_rule_manager()
		custom_rules_text = rule_manager.format_rules_for_prompt()
		if custom_rules_text:
			final_prompt += "\n" + custom_rules_text
	
	return final_prompt
