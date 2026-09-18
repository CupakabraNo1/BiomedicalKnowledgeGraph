"""Cypher queries used by data.py.

Labels and relationship types cannot be passed as parameters in Cypher, so the
queries that need one are functions; the rest are plain constants.
"""

AWAIT_INDEXES = "CALL db.awaitIndexes()"

COUNT_NODES = "MATCH (n) RETURN count(n) AS count"
COUNT_EDGES = "MATCH ()-[r]->() RETURN count(r) AS count"

DELETE_EDGES = "MATCH ()-[r]->() CALL { WITH r DELETE r } IN TRANSACTIONS OF 10000 ROWS"
DELETE_NODES = "MATCH (n) CALL { WITH n DELETE n } IN TRANSACTIONS OF 10000 ROWS"

FETCH_NODES = "MATCH (n) RETURN labels(n)[0] AS kind, properties(n) AS props"

FETCH_EDGES = """
MATCH (a)-[r]->(b)
RETURN labels(a)[0] AS src_kind, a.identifier AS src,
       labels(b)[0] AS tgt_kind, b.identifier AS tgt,
       type(r) AS reltype, properties(r) AS props
"""


def create_constraint(kind):
    """Unique identifier per label. The index behind it keeps MERGE fast."""
    return f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{kind}`) REQUIRE n.identifier IS UNIQUE"


def merge_nodes(kind):
    return f"""
    UNWIND $rows AS row
    MERGE (n:`{kind}` {{identifier: row.identifier}})
    SET n += row.props
    """


def merge_edges(src_kind, reltype, tgt_kind):
    return f"""
    UNWIND $rows AS row
    MATCH (a:`{src_kind}` {{identifier: row.src}})
    MATCH (b:`{tgt_kind}` {{identifier: row.tgt}})
    MERGE (a)-[r:`{reltype}`]->(b)
    SET r += row.props
    """
