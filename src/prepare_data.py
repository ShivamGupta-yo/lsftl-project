import re
import random

random.seed(42)  # ensures the same random subset every time we run this

def clean_line(line):
    line = line.strip()
    # Remove invisible Unicode control/formatting characters (LRM, RLM, ZWJ, ZWNJ, etc.)
    line = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', line)
    # Remove spaced-out letter ads like "- t h a t o -"
    line = re.sub(r'-\s*[a-zA-Z]\s*(?:[a-zA-Z]\s*){2,}-', '', line)
    # Remove lines mentioning uploader/subtitle credits
    line = re.sub(r'\b(subtitle|download|sync|corrected|by)\b.*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\s+', ' ', line).strip()
    return line

def is_mostly_devanagari(text, threshold=0.5):
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = sum(1 for c in text if c.isalpha())
    if total_chars == 0:
        return False
    return (devanagari_chars / total_chars) >= threshold


def load_and_clean(hi_path, ms_path, max_pairs=20000, min_len=3, max_len=100):
    with open(hi_path, encoding='utf-8') as f:
        hi_lines = f.readlines()
    with open(ms_path, encoding='utf-8') as f:
        ms_lines = f.readlines()

    pairs = []
    for hi, ms in zip(hi_lines, ms_lines):
        hi_c, ms_c = clean_line(hi), clean_line(ms)
        if min_len <= len(hi_c.split()) <= max_len and min_len <= len(ms_c.split()) <= max_len:
            if is_mostly_devanagari(hi_c):
                pairs.append((hi_c, ms_c))

    random.shuffle(pairs)
    return pairs[:max_pairs]

if __name__ == "__main__":
    pairs = load_and_clean(
        "/raid/home/loitongbam/Shivam-PhD/lsftl-project/data/raw/OpenSubtitles.hi-ms.hi",
        "/raid/home/loitongbam/Shivam-PhD/lsftl-project/data/raw/OpenSubtitles.hi-ms.ms",
        max_pairs=20000
    )

    print(f"Kept {len(pairs)} clean pairs")
    
    if pairs:
        print("Sample:", pairs[0])

    with open(
        "/raid/home/loitongbam/Shivam-PhD/lsftl-project/data/raw/clean_hi.txt",
        "w",
        encoding="utf-8"
    ) as f_hi, \
         open(
        "/raid/home/loitongbam/Shivam-PhD/lsftl-project/data/raw/clean_ms.txt",
        "w",
        encoding="utf-8"
    ) as f_ms:

        for hi, ms in pairs:
            f_hi.write(hi + "\n")
            f_ms.write(ms + "\n")