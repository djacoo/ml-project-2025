# Manual correction dictionary (Override)
COUNTRY_OVERRIDES = {
    # --- Fix for incorrect splits (commas in official names) ---
    'plurinational state of': 'Bolivia',
    'bolivarian republic of': 'Venezuela',
    'republic of korea': 'South Korea',
    
    # --- North America ---
    'usa': 'United States', 'us': 'United States', 'en:us': 'United States', 
    'états-unis': 'United States',
    
    # --- Europe (Common translations) ---
    'uk': 'United Kingdom', 'en:gb': 'United Kingdom', 'royaume-uni': 'United Kingdom',
    'de': 'Germany', 'deutschland': 'Germany', 'Deutschland': 'Germany', 'allemagne': 'Germany',
    'fr': 'France', 'en:fr': 'France', 'francia': 'France',
    'es': 'Spain', 'españa': 'Spain', 'espagne': 'Spain',
    'it': 'Italy', 'italia': 'Italy',
    'nl': 'Netherlands', 'nederland': 'Netherlands', 'holland': 'Netherlands',
    'be': 'Belgium', 'belgique': 'Belgium',
    'ch': 'Switzerland', 'suisse': 'Switzerland', 'schweiz': 'Switzerland',
    'cz': 'Czechia', 'česko': 'Czechia',
    'pl': 'Poland', 'polska': 'Poland',
    'se': 'Sweden', 'sverige': 'Sweden',
    'no': 'Norway', 'norge': 'Norway',
    
    # --- Rest of the world ---
    'ru': 'Russian Federation', 'russia': 'Russian Federation',
    'jp': 'Japan',
    'cn': 'China',
    'br': 'Brazil', 'brasil': 'Brazil',
}