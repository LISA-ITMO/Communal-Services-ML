from sqlalchemy import create_engine
import pandas as pd
import os
import sys


PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, '..'))

# Connection details
DB_NAME = "postgres"
USER = "postgres"
PASSWORD = "goshective"
HOST = "localhost"
PORT = "5432"

# Create a connection
engine = create_engine(f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}")

# Define the query
query = """
SELECT  
"Код_тематики" AS topic_id,
"Президент_тематика" AS president_topic,
"Тематика" AS detailed_topic,
"Содержание_обращения" AS appeal 
FROM public."SED"
"""

df = pd.read_sql(query, engine)

# Close the connection
engine.dispose()


print(df.head())

df.to_csv(os.path.join(PATH, 'Database', 'db_2.csv'), index=False, sep=';')