import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# Задаємо параметри
n = 147
mu = 0  # математичне сподівання
sigma = 1.4  # стандартне відхилення

# Генеруємо вибірку
np.random.seed(36)  # встановлюємо seed для відтворюваності результатів
sample = np.random.normal(mu, sigma, n)


# Розрахунки
def mean(sample):
    total_sum = sum(sample)
    count = len(sample)
    return total_sum / count


def median(sample):
    sorted_sample = sorted(sample)
    length = len(sorted_sample)
    middle = length // 2

    if length % 2 == 0:  # парна кількість елементів
        return (sorted_sample[middle - 1] + sorted_sample[middle]) / 2
    else:  # непарна кількість елементів
        return sorted_sample[middle]

def mode(sample):
    sample_list = list(sample)
    unique_samples = set(sample_list)
    frequencies = []

    for s in unique_samples:
        freq = sample_list.count(s)
        frequencies.append((s, freq))

    max_freq = 0
    for s, freq in frequencies:
        if freq > max_freq:
            max_freq = freq

    if max_freq == 1:
        return "Моди вибірки немає"

    mode = []
    for s, freq in frequencies:
        if freq == max_freq:
            mode.append(s)
    return mode

def variance(sample):
    n = len(sample)
    mean = sum(sample) / n
    var = sum((x - mean) ** 2 for x in sample) / n
    return var

def standart_dev(sample):
    return math.sqrt(variance(sample))

# Виводимо результати
print(f"Вибіркове середнє: {mean(sample)}")
print(f"Медіана: {median(sample)}")
print(f"Мода: {mode(sample)}")
print(f"Вибіркова дисперсія: {variance(sample)}")
print(f"Вибіркове середньоквадратичне відхилення: {standart_dev(sample)}")

# Будуємо графіки
plt.figure(figsize=(18, 12))

# Гістограма частот
plt.subplot(231)
plt.hist(sample, bins='auto', color='orange', alpha=0.7, rwidth=0.85)
plt.title('Гістограма частот')

# Полігон частот
plt.subplot(232)
sns.kdeplot(sample, color='green')
plt.title('Полігон частот')

# Діаграма розмаху
plt.subplot(234)
sns.boxplot(sample, color='purple')
plt.title('Діаграма розмаху')

# Кругова діаграма
plt.subplot(235)
bins = pd.cut(sample, bins=10)  # розбиваємо дані на 10 інтервалів
labels, counts = np.unique(bins, return_counts=True)
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
plt.title('Кругова діаграма')

# Діаграма Парето
plt.subplot(236)
counts = bins.value_counts().sort_values(ascending=False)
counts.plot(kind='bar', color='darkred')
plt.title('Діаграма Парето')

plt.tight_layout()
plt.show()

