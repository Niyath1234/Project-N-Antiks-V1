import sqlite3
from typing import List, Tuple, Any, Optional

DANGEROUS_KEYWORDS = {
    'ATTACH', 'DETACH', 'PRAGMA', 'VACUUM', 'ALTER', 'DROP', 'TRIGGER', 'INDEX',
    'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'GRANT', 'REVOKE'
}


def create_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sales (
            product_id TEXT,
            product_name TEXT,
            product_category TEXT,
            product_subcategory TEXT,
            product_price REAL,
            product_quantity INTEGER,
            product_total_sales REAL
        );
        """
    )
    conn.commit()


def create_compat_views(conn: sqlite3.Connection) -> None:
    # Create a TEMP compatibility view that exposes commonly expected aliases
    # This lets queries like SUM(sales_amount) FROM sales work out-of-the-box
    conn.execute(
        """
        CREATE TEMP VIEW IF NOT EXISTS sales AS
        SELECT
            product_id,
            product_name,
            product_category,
            product_category AS category,
            product_subcategory,
            product_price,
            product_quantity,
            product_total_sales,
            product_total_sales AS total_sales,
            product_total_sales AS sales_amount
        FROM main.sales;
        """
    )
    conn.commit()


def seed_example_rows(conn: sqlite3.Connection) -> None:
    rows = [
        ("P001", "Widget A", "Gadgets", "Small", 10.0, 5, 50.0),
        ("P002", "Widget B", "Gadgets", "Large", 20.0, 3, 60.0),
        ("P003", "Sprocket X", "Hardware", "Metal", 15.0, 4, 60.0),
        ("P004", "Sprocket Y", "Hardware", "Plastic", 8.0, 10, 80.0),
    ]
    conn.executemany(
        """
        INSERT INTO main.sales (
            product_id, product_name, product_category, product_subcategory,
            product_price, product_quantity, product_total_sales
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def register_numeric_udfs(conn: sqlite3.Connection) -> None:
    """Register regex-like numeric extraction and cleaning helpers for dirty text values.
    Functions added:
    - EXTRACT_NUMBER(text): returns first -?\d+(?:\.\d+)? match or NULL
    - TO_REAL_CLEAN(text): float(EXTRACT_NUMBER(text)) or NULL
    - NUMERIC_ONLY(text): keep only digits, minus, and dot (best-effort)
    - REGEXP_REPLACE(text, pattern, repl): Python re.sub wrapper
    - TRY_CAST_REAL(text): float(text) or NULL if fails
    - CLEAN_NUMERIC(text): strip commas, currency, %, then float or NULL
    """
    import re

    def extract_number(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        s = str(val)
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return m.group(0) if m else None

    def to_real_clean(val: Optional[str]) -> Optional[float]:
        num = extract_number(val)
        try:
            return float(num) if num is not None else None
        except Exception:
            return None

    def numeric_only(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        out = ''.join(ch for ch in str(val) if ch.isdigit() or ch in '.-')
        return out if out else None

    def regexp_replace(text: Optional[str], pattern: Optional[str], repl: Optional[str]) -> Optional[str]:
        if text is None or pattern is None or repl is None:
            return text
        try:
            return re.sub(pattern, repl, str(text))
        except Exception:
            return text

    def try_cast_real(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    def clean_numeric(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        s = str(val)
        s = re.sub(r"[,$₹£€%]", "", s).strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            # last resort: extract first number
            return to_real_clean(s)

    conn.create_function("EXTRACT_NUMBER", 1, extract_number)
    conn.create_function("TO_REAL_CLEAN", 1, to_real_clean)
    conn.create_function("NUMERIC_ONLY", 1, numeric_only)
    conn.create_function("REGEXP_REPLACE", 3, regexp_replace)
    conn.create_function("TRY_CAST_REAL", 1, try_cast_real)
    conn.create_function("CLEAN_NUMERIC", 1, clean_numeric)


def is_safe_sql(sql: str) -> bool:
    """Check if SQL is safe (read-only SELECT/WITH, no DDL/DML)."""
    if not sql:
        return False
    
    # Strip ALL comments (leading, inline, trailing)
    s = sql.strip()
    
    # Remove block comments
    while '/*' in s:
        start = s.find('/*')
        end = s.find('*/', start)
        if end == -1:
            break
        s = s[:start] + ' ' + s[end+2:]
    
    # Remove line comments
    lines = s.split('\n')
    cleaned_lines = []
    for line in lines:
        if '--' in line:
            line = line[:line.index('--')]
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    s_clean = ' '.join(cleaned_lines).strip()
    
    if not s_clean:
        return False
    
    # Remove trailing semicolon for checking
    s_clean = s_clean.rstrip(';').strip()
    
    # Check for multiple statements (semicolons in the middle)
    if ';' in s_clean:
        return False  # Multi-statement SQL
    
    # Convert to uppercase for keyword checking
    normalized = s_clean.upper()
    
    # MUST start with SELECT or WITH
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return False
    
    # Block dangerous keywords (DDL/DML)
    for kw in DANGEROUS_KEYWORDS:
        # Word boundary check
        if f' {kw} ' in f' {normalized} ':
            return False
    
    # All SELECT/WITH queries are allowed!
    return True


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description] if cur.description else []
    return col_names, rows


def format_table(headers: List[str], rows: List[Tuple[Any, ...]]) -> str:
    if not headers:
        return "(no result set)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    def fmt_row(cells: List[Any]) -> str:
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for r in rows:
        lines.append(fmt_row(list(r)))
    return "\n".join(lines)
