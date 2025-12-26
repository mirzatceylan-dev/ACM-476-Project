import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# VERİ OKUMA
df = pd.read_csv("medical_insurance.csv")
df = df.drop(['person_id'], axis=1, errors='ignore')

# İSİMLENDİRME
# Medical insurance veri setinde 'charges' genellikle en son sütundur.
if 'charges' in df.columns:
    df = df.rename(columns={'charges': 'Target'})
else:
    # Eğer isim farklıysa son sütunu Target yap (emin olmak için bu şekilde yaptım)
    df.rename(columns={df.columns[-1]: 'Target'}, inplace=True)

# Kategorik verileri sayıya çevirdim
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Sadece Target dışındaki sütunları V1, V2... yaptım
cols = [c for c in df.columns if c != 'Target']
df = df.rename(columns={old: f'V{i+1}' for i, old in enumerate(cols)})

# 'Target' kelimesini listede görmek için kontrol ediyoruz
print("Güncel Sütunlar:", df.columns)

#ADIM 1: TANIMLAYICI İSTATİSTİKLER
num_var = [col for col in df.columns if df[col].dtype != 'O']
desc_agg = ['sum', 'mean', 'max', 'min', 'std', 'var']
desc_summ = df[num_var].agg(desc_agg)

print("\n--- İstatistiksel Özet ---")
print(desc_summ.iloc[:, :5]) #Kolaylık olması açısından ilk 5 sütunu aldım


#ADIM 2: HEDEF DEĞİŞKEN ANALİZİ

# Sütun isimlerini kontrol edelim, Target en sonda olmalı
target_col = 'Target'

if target_col in df.columns:
    print("\n--- Target Dağılım Analizi ---")
    mean_val = df[target_col].mean()

    # Büyüktür/küçüktür sayımı yaptım
    above_mean = df[df[target_col] > mean_val][target_col].count()
    below_mean = df[df[target_col] < mean_val][target_col].count()

    print(f"Ortalama: {mean_val:.2f}")
    print(f"Ortalamanın üzerindeki kayıt sayısı: {above_mean}")
    print(f"Ortalamanın altındaki kayıt sayısı: {below_mean}")

    #Burda verinin dengesine bakıyoruz
    if above_mean > below_mean:
        print("Yorum: Veri sağa çarpık (pozitif skew). Çoğu veri ortalamanın altında toplanmış.")
    else:
        print("Yorum: Veri sola çarpık (negatif skew).")
else:
    print(f"Hata: {target_col} sütunu bulunamadı! Mevcut sütunlar: {df.columns}")

# Önce sadece V1 için Boxplot çizimi yaptım
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['V1'])
plt.title("V1 Değişkeni Boxplot (Aykırı Değer Kontrolü)")
plt.show()

# num_summary e başladım burda
#ADIM 3: HİSTOGRAM VE BOXPLOT DÖNGÜSÜ
def num_summary(data, numerical_col, plot=True):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"\n--- {numerical_col} İstatistikleri ---")
    print(data[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        data[numerical_col].hist(bins=20, edgecolor='black')
        plt.title(f"{numerical_col} Histogram")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[numerical_col])
        plt.title(f"{numerical_col} Boxplot")

        plt.show(block=True)

print("\nTÜM değişkenler için detaylı grafikler hazırlanıyor...")

# Target ve Cluster hariç tüm sütunları listeye alıyorum
tum_degiskenler = [col for col in df.columns if col not in ['Target', 'cluster']]

for col in tum_degiskenler:
    num_summary(df, col, plot=True)


#ADIM 4: KORELASYON ANALİZİ VE SÜTUN ELEME

# Korelasyon matrisini hesaplayalım
corr = df.corr().abs()

# Isı Haritası (Heatmap) Çizdim burda
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), cmap='RdBu', annot=False)
plt.title("Değişkenler Arası Korelasyon Analizi (Isı Haritası)")
plt.show()

upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.90)]

print("\n--- Çok Yüksek Korelasyonlu (Silinecek) Sütunlar ---")
if len(drop_list) > 0:
    print(f"{len(drop_list)} adet kopya sütun bulundu: {drop_list}")
    df = df.drop(drop_list, axis=1)
    print(f"Yeni Veri Boyutu: {df.shape}")
else:
    print("Birbirine çok benzeyen (kopya) sütun bulunamadı.")

#Overfitting riskini azaltmak için yaptım.
#Eğer iki değişken birbirinin %90 aynısıysa, model gereksiz yere iki kat fazla işlem yapar.
#Birini silmek modeli daha kararlı hale getirir.

#ADIM 5: K-MEANS KÜMELEME (CLUSTERING)
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

#Standartlaştırma (StandardScaler)
# Veriler farklı ölçeklerde olduğu için hepsini aynı seviyeye çekiyorum
X = df.drop('Target', axis=1)
X_scaled = StandardScaler().fit_transform(X)

# Optimum Küme Sayısını Belirleme (Elbow)
print("\nOptimum küme sayısı (Dirsek Noktası) hesaplanıyor...")
kmeans_model = KMeans(random_state=17)
visualizer = KElbowVisualizer(kmeans_model, k=(2, 10))
visualizer.fit(X_scaled)
visualizer.show()

# Grafikte bulunan en iyi sayıya (elbow_value_) göre gruplandırıyorum
optimal_k = visualizer.elbow_value_ if visualizer.elbow_value_ else 4
final_kmeans = KMeans(n_clusters=optimal_k, random_state=17).fit(X_scaled)

# Her satıra hangi grupta olduğu bilgisini (Cluster ID) ekliyoruz
df['cluster'] = final_kmeans.labels_ + 1

print(f"\nİşlem Başarıyla Tamamlandı!")
print(f"Veri {optimal_k} ana gruba (küme) ayrıldı.")
print("\nKümelere Göre İlk 5 Satır:")
print(df[['V1', 'V2', 'Target', 'cluster']].head())

#ADIM 6: TARGET ANALİZİ VE PCA
print("\n--- Target Değişkenine Göre Özelliklerin Ortalaması (TÜM DEĞİŞKENLER İÇİN!) ---")

#Fonksiyonu tekrar garantiye alalım
def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target)[num_col].mean(), end='\n\n')

tum_sutunlar = [col for col in df.columns if col not in ['Target', 'cluster']]

print(f"[BİLGİ] Toplam {len(tum_sutunlar)} adet değişken analiz ediliyor...")

for col in tum_sutunlar:
    print(f"### {col} Analizi ###")
    target_summary_with_num(df, 'Target', col)
    print("-" * 30)

from sklearn.decomposition import PCA

print("\n--- PCA (Principal Component Analysis) Analizi ---")
pca = PCA()
pca_fit = pca.fit_transform(X_scaled)

# Açıklanan varyans oranı
print("Açıklanan Varyans Oranları (İlk 5 bileşen):")
print(pca.explained_variance_ratio_[:5])


# PCA KÜME HARİTASI OLUŞTURDUM (2D Scatter Plot)
print("\nPCA Küme Haritası çiziliyor...")
plt.figure(figsize=(12, 8))

# PCA'nın ilk 2 bileşenini alıyorum (X ve Y ekseni yapmak için)
pca_2d = pca_fit[:, :2]

# Scatter plot çizdim ve her kümeyi farklı renk ayarladım
scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster ID')
plt.title(f"Müşteri Segmentasyon Haritası (PCA ile 2 Boyuta İndirgenmiş)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("\nKümelerin Target (Maliyet) Ortalamaları çiziliyor...")
plt.figure(figsize=(10, 6))

#hue='cluster' ve legend=False ekledim çünkü grafikler çalışıyordu ama consolda hata alıyordum
sns.barplot(x='cluster', y='Target', hue='cluster', data=df, palette='viridis', legend=False)

plt.title("Hangi Küme Daha Maliyetli?", fontsize=14)
plt.xlabel("Müşteri Grubu (Cluster)")
plt.ylabel("Ortalama Target (Maliyet)")

# Çubukların üzerine sayılarını yazdım
ax = plt.gca()
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.show()

#Sonuçları CSV olarak kaydettim
df.to_csv('proje_final_sonuclari.csv', index=False)
