"""
Modul untuk fusi hasil klasifikasi teks dan citra.
"""

def fuse_results(text_logit, image_logit):
    """
    Menggabungkan hasil klasifikasi teks dan citra menggunakan operator AND.

    Aturan fusi (Logika AND):
    - Hasilnya 1 (Bukan Judi) hanya jika kedua input adalah 1.
    - Dalam semua kasus lainnya, hasilnya adalah 0 (Judi).

    Args:
        text_logit (int): Logit hasil klasifikasi teks (0 atau 1).
        image_logit (int): Logit hasil klasifikasi citra (0 atau 1).

    Returns:
        int: Logit akhir hasil fusi (0 atau 1).
    
    Raises:
        ValueError: Jika input bukan 0 atau 1.
    """
    if text_logit not in [0, 1] or image_logit not in [0, 1]:
        raise ValueError("Input logit harus berupa 0 atau 1.")
        
    # Logika AND: hasil 1 hanya jika kedua input adalah 1.
    final_logit = text_logit and image_logit
    
    return final_logit
