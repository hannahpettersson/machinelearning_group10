# import pandas as pd
# import matplotlib.pyplot as plt


# df = pd.read_csv('diabetic_data.csv')
# df.info()

# # kolumnerna 22 till 46 (index-baserat)
# med_cols = df.columns[20:44]

# # 3. Räkna hur många som "tar" medicinen 
# # Vi räknar alla rader som INTE har värdet 'No', 'None' eller är tomma (NaN)
# counts = {}
# for col in med_cols:
#     taking_med = df[~df[col].isin(['No', 'None']) & df[col].notna()]
#     counts[col] = len(taking_med)

# # Omvandla till en Series och sortera för bättre överblick
# med_counts = pd.Series(counts).sort_values(ascending=False)

# # 4. Skriv ut resultatet
# print("Antal patienter per medicin/test:")
# print(med_counts)

# # 5. Skapa ett diagram för de 20 vanligaste
# med_counts.head(26).plot(kind='bar', figsize=(12, 6), color='skyblue', edgecolor='black')
# plt.title('Antal patienter per medicin/test (Exklusive No/None)')
# plt.ylabel('Antal patienter')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 1. Läs in filen
df = pd.read_csv('diabetic_data.csv')

# 2. Välj kolumnerna 22 till 46
med_cols = df.columns[22:46]

# 3. Räkna användare i den ordning kolumnerna dyker upp
counts = []
labels = []

for col in med_cols:
    # Vi räknar alla som INTE har 'No', 'None' eller NaN
    count = df[~df[col].isin(['No', 'None']) & df[col].notna()].shape[0]
    counts.append(count)
    labels.append(col)

# 4. Skapa diagrammet
plt.figure(figsize=(15, 7))
plt.bar(labels, counts, color='mediumseagreen', edgecolor='black')

# Inställningar för att göra det läsbart
plt.title('Antal patienter per kolumn (i ordning 22-46)', fontsize=14)
plt.ylabel('Antal patienter')
plt.xlabel('Medicin / Test')
plt.xticks(rotation=90) # Rotera 90 grader så namnen inte krockar
plt.grid(axis='y', linestyle='--', alpha=0.7) # Hjälplinjer för lättare avläsning

plt.tight_layout()
plt.show()

# 5. Skriv ut listan i terminalen också
total_patients = len(df)

for i, col in enumerate(labels):
    percentage = (counts[i] / total_patients) * 100
    print(f"Index {i+22}: {col} -> {counts[i]} patienter och i procent: {percentage:.2f}%")