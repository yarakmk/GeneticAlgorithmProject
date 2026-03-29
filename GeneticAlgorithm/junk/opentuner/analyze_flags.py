import csv
from collections import Counter

counter = Counter()

with open("results/best_configs.csv") as f:
    for row in csv.DictReader(f):
        for flag, setting in row.items():
            if setting == "on":
                counter[flag] += 1

# take top 50 most frequently "on" flags
top50 = counter.most_common(50)

with open("best_flags_ranked.csv", "w") as out:
    writer = csv.writer(out)
    writer.writerow(["flag","frequency"])
    for flag, freq in top50:
        writer.writerow([flag, freq])
