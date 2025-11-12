from database import engine, Base
import models  # Ensure tables are recognized
from sqlalchemy import inspect

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")

# Inspect tables to confirm they exist
inspector = inspect(engine)
tables = inspector.get_table_names()

if tables:
    print("Existing tables:", tables)
else:
    print("No tables found!")
