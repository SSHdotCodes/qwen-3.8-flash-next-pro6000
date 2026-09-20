import hashlib,json
def apply_migrations(connection,migrations):
    if connection.in_transaction: raise ValueError('active transaction')
    records=list(migrations)
    versions=[v for v,_ in records]
    if any(type(v) is not int or v<=0 for v in versions) or len(set(versions))!=len(versions):
        raise ValueError('invalid or duplicate versions')
    records.sort(key=lambda x:x[0])
    def checksum(statements):
        return hashlib.sha256(json.dumps(statements,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
    applied=[]
    connection.execute('BEGIN IMMEDIATE')
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS _migrations(version INTEGER PRIMARY KEY,checksum TEXT NOT NULL)')
        existing=dict(connection.execute('SELECT version,checksum FROM _migrations'))
        for v,statements in records:
            if v in existing and existing[v]!=checksum(statements):raise ValueError('checksum changed')
        for v,statements in records:
            if v in existing:continue
            for statement in statements:
                connection.execute(statement)
            connection.execute('INSERT INTO _migrations VALUES(?,?)',(v,checksum(statements)))
            applied.append(v)
        connection.commit()
        return applied
    except BaseException:
        connection.rollback()
        raise
