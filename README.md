# System rozpoznawania cyfr odręcznych z wykorzystaniem klasyfikatora k‑NN i interfejsu Tkinter

Interaktywny system OCR oparty na klasyfikatorze **k‑Nearest Neighbors** z graficznym interfejsem użytkownika. Użytkownik rysuje cyfrę na płótnie, a system przeprowadza pełny proces przetwarzania obrazu, klasyfikacji oraz wizualizacji wyników.

## Spis treści

- [Wymagania systemowe](#wymagania-systemowe)
- [Instalacja](#instalacja)
- [Uruchomienie](#uruchomienie)
- [Zbiór danych](#zbiór-danych)
- [Architektura systemu](#architektura-systemu)
  - [Interfejs użytkownika](#interfejs-użytkownika)
  - [Przetwarzanie obrazu](#przetwarzanie-obrazu)
  - [Klasyfikacja](#klasyfikacja)
  - [Wizualizacje](#wizualizacje)
- [Struktura projektu](#struktura-projektu)
- [Możliwe kierunki rozwoju](#możliwe-kierunki-rozwoju)

---

## Wymagania systemowe

- **Python** 3.10–3.12
- **Tkinter** (część standardowej dystrybucji Pythona)

### Zależności

| Biblioteka     | Wersja |
| -------------- | ------ |
| `numpy`        | 1.26.4 |
| `Pillow`       | 10.2.0 |
| `scikit-learn` | 1.4.2  |
| `matplotlib`   | 3.8.3  |

---

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python main.py
```

1. Narysuj cyfrę (0–9) na białym płótnie za pomocą myszy.
2. Wprowadź wartość parametru **k** (liczba najbliższych sąsiadów).
3. Kliknij **Predict**, aby przeprowadzić klasyfikację.
4. Kliknij **Clear**, aby wyczyścić płótno i rozpocząć od nowa.

---

## Zbiór danych

System wykorzystuje zbiór **Digits** z modułu `sklearn.datasets`:

| Właściwość              | Wartość     |
| ----------------------- | ----------- |
| Liczba próbek           | 1797        |
| Rozdzielczość obrazu    | 8×8 pikseli |
| Zakres wartości pikseli | 0–16        |
| Liczba klas             | 10 (0–9)    |

Zbiór stanowi referencyjną bazę uczącą dla klasyfikatora k‑NN.

---

## Architektura systemu

### Interfejs użytkownika

Zbudowany w oparciu o **Tkinter**, składa się z trzech głównych paneli:

```
┌──────────────────────────────────────────────────────┐
│                     KNN OCR                          │
├───────────┬───────────────────────┬──────────────────┤
│           │                       │                  │
│  Płótno   │  Wizualizacje         │  Histogram       │
│  rysunku  │  • 32×32 preview      │  głosowania      │
│  300×300  │  • średnia sąsiadów   │  k‑NN            │
│           │  • mapa różnic        │                  │
│           │                       │                  │
├───────────┴───────────────────────┴──────────────────┤
│  [k value: ___]  [Clear]  [Predict]                  │
│  Prediction: ?                                       │
└──────────────────────────────────────────────────────┘
```

- **Panel rysowania** — obszar 300×300 px do odręcznego wprowadzania cyfr.
- **Panel wizualizacji** — trzy wykresy: przeskalowany obraz 32×32, średnia obrazów _k_ najbliższych sąsiadów oraz mapa różnic między rysunkiem a średnią sąsiadów.
- **Panel głosowania** — histogram rozkładu głosów _k_ najbliższych sąsiadów dla poszczególnych klas.

### Przetwarzanie obrazu

Pipeline przetwarzania danych wejściowych:

```
Rysunek 300×300
       ↓
  Skalowanie (Lanczos) → 32×32
       ↓
  Inwersja kolorów
       ↓
  Progowanie (próg = 50)
       ↓
  Redukcja 32×32 → 8×8 (bloki 4×4)
       ↓
  Spłaszczenie → wektor cech (64 wymiary)
```

Każdy etap zapewnia zgodność formatu danych wejściowych z reprezentacją zbioru Digits.

### Klasyfikacja

| Parametr            | Wartość                     |
| ------------------- | --------------------------- |
| Algorytm            | k‑NN                        |
| Metryka odległości  | Manhattan (L1)              |
| Metoda wyszukiwania | brute force                 |
| Liczba sąsiadów (k) | określana przez użytkownika |

### Wizualizacje

System generuje trzy kluczowe wizualizacje:

1. **Średnia najbliższych sąsiadów**

   $$A = \frac{1}{k} \sum_{i=1}^{k} N_i$$

   Obraz będący średnią arytmetyczną _k_ najbliższych sąsiadów w przestrzeni 32×32.

2. **Mapa różnic**

   $$D = |A - I|$$

   Bezwzględna różnica między rysunkiem użytkownika a średnią sąsiadów — pozwala zidentyfikować piksele, które najbardziej odbiegają od wzorca.

3. **Histogram głosowania**

   Rozkład etykiet wśród _k_ najbliższych sąsiadów — umożliwia ocenę pewności klasyfikacji.

---

## Struktura projektu

```
.
├── main.py              # Kod źródłowy aplikacji
├── requirements.txt     # Lista zależności
└── README.md            # Dokumentacja projektu
```

---

## Możliwe kierunki rozwoju

- Implementacja dodatkowych klasyfikatorów (SVM, sieci neuronowe)
- Wprowadzenie filtrów wstępnych (np. wygładzanie Gaussa)
- Możliwość trenowania na danych użytkownika
- Eksport raportów klasyfikacji
- Integracja z modelami głębokiego uczenia

---

## Podsumowanie

Projekt stanowi kompletny, interaktywny system OCR oparty na klasyfikatorze k‑NN. Dzięki rozbudowanym wizualizacjom użytkownik może analizować proces klasyfikacji na poziomie struktury pikselowej, co czyni system wartościowym narzędziem dydaktycznym i eksperymentalnym. Aplikacja łączy klasyczne metody uczenia maszynowego z intuicyjnym interfejsem graficznym, umożliwiając praktyczne badanie właściwości algorytmu k‑NN.
