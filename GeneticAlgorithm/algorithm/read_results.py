import sqlite3
import zlib
import pickle

conn = sqlite3.connect('opentuner.db/192.168.1.126.db')
cursor = conn.cursor()

# Count completed trials
cursor.execute("SELECT COUNT(*) FROM result WHERE time IS NOT NULL")
print(f"Completed trials: {cursor.fetchone()[0]}")

# Get the best result and its configuration
cursor.execute('''
    SELECT c.data, r.time 
    FROM result r
    JOIN configuration c ON r.configuration_id = c.id
    WHERE r.time IS NOT NULL
    ORDER BY r.time ASC
    LIMIT 1
''')

row = cursor.fetchone()
config_data, best_time = row

decompressed = zlib.decompress(config_data)
config = pickle.loads(decompressed)

print(f"\n{'='*60}")
print(f"  BEST HYPERPARAMETERS FOUND")
print(f"  Best fitness : {best_time:.4f}s")
print(f"{'='*60}")
for param, value in config.items():
    print(f"  {param:<25} : {value}")
print(f"{'='*60}")

conn.close()