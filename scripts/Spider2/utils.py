import sqlite3

def extract_schema_with_keys(db_path):
    """Extract schema information from a SQLite database,
    including table definitions, primary keys, and foreign keys."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    out = []

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]

    for table in tables:
        # Column definitions
        cursor.execute(f"PRAGMA table_info('{table}')")
        cols = cursor.fetchall()
        col_defs = [f"{c[1]} ({c[2]})" for c in cols]
        out.append(f"Table {table}: {', '.join(col_defs)}")

        # Primary key
        pk_cols = [c[1] for c in cols if c[5]]
        if pk_cols:
            out.append(f"Primary Key: {', '.join(pk_cols)}")

        # Foreign keys
        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        fks = cursor.fetchall()
        for fk in fks:
            out.append(f"Foreign Key: {fk[3]} references {fk[2]}({fk[4]})")

    conn.close()
    return "\n".join(out)
