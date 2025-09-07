Zde je přeložené a upravené README do češtiny pro tvůj bakalářský projekt:

---

# Optimalizace sdílení energie v komunitě

Tento repozitář obsahuje kód a modely vytvořené pro bakalářskou práci zaměřenou na optimalizaci redistribuce energie v rámci energetické komunity. Hlavním cílem je minimalizovat celkové nealokované množství energie (nevyužitá nabídka výrobců a nenaplněná poptávka spotřebitelů) za období jednoho měsíce v 15minutových intervalech.

## Struktura projektu

* **Analysis/**
  Notebooky pro průzkum a analýzu dat.
* **Models/**
  Jádro optimalizačních modelů a pomocných nástrojů:

  * `dynamic_weights.py`, `static_weights.py`: Modely pro optimalizaci vah.
  * `heuristics.py`: Heuristické algoritmy pro optimalizaci preferencí.
  * `experiments.py`, `testing.ipynb`: Skripty pro testování a experimenty.
  * `utils.py`: Pomocné funkce.
  * `whole_model.py`: Hlavní model kombinující všechny komponenty. Nekonvexní formulace.
* **Outputs/**
  Výstupy experimentů, grafy a výsledkové soubory.

## Přehled problému

Cílem je efektivně přidělovat energii od výrobců ke spotřebitelům při respektování následujících omezení:

* Každý spotřebitel může být napojen maximálně na 5 výrobců.
* Preference (priority) jsou zadány jako celá čísla (5 = nejvyšší priorita, 1 = nejnižší, 0 = žádné spojení).
* Alokace nesmí překročit nabídku výrobců ani poptávku spotřebitelů.

### Požadavky modelu

* **Vstupy:**

  * Časové řady výroby a spotřeby energie (15minutové intervaly).
  * Matice preferencí (priority spotřebitel-výrobce).
* **Výstupy:**

  * Realizovatelné matice vah a priorit.
  * Celkové nealokované množství energie.
  * Rozklad na nenaplněnou poptávku a nevyužitou nabídku.

## Použité přístupy

### Optimalizace vah

* **Statické a dynamické modely:**
  Hledání optimálních vah pro rozdělení energie mezi výrobce a spotřebitele při fixních preferencích.
* **Přístupy:**

  * Vysoce přesné (nekonvexní, pomalé, přesné).
  * Zjednodušené konvexní (rychlé, přibližné).

### Optimalizace preferencí

* **Heuristické algoritmy:**

  * Genetický algoritmus (GA)
  * Simulované žíhání (SA)
  * Lokální prohledávání (Hill Climbing)
* Tyto metody hledají strukturu priorit, která minimalizuje nealokovanou energii.

## Použití

1. **Příprava dat:**
   Zpracovaná data umístěte do adresáře `data/`.
2. **Spuštění modelů:**
   Spusťte optimalizační skripty z adresáře `Models/`. Například:

   ```sh
   python Models/dynamic_weights.py
   ```

---

Pokud chceš, mohu README také naformátovat podle akademického stylu nebo přidat úvodní odstavec s odkazem na tvou bakalářskou práci.

