from collections import deque
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import re

# Inisialisasi variabel global
metode_ocr = {
    "original": "",
    "negative": "",
    "inverted": ""
}

def text_conversion_ocr_capital(text):
    char_map = [
        (2, 'I', '1'), (2, 'T', '1'), (1, '4', 'H'),
        (9, '6', 'G'), (9, '0', 'J'), (3, '0', 'J'),
        (10, 'Y', 'K'), (10, 'X', 'K'), (12, 'G', '6'),
    ]
    text = text.upper().replace(' ', '')
    return convert_chars_at_positions(text, char_map)

def karakter_tidak_valid(image, sementara):
    (T, thresh) = cv2.threshold(image, 77, 255, cv2.THRESH_BINARY)
    reader = easyocr.Reader(['id'])
    results = reader.readtext(thresh)
    array_text = [text for (_, text, _) in results]
    all_text = ' '.join(array_text[::-1])
    sementara = text_conversion_ocr_capital(sementara)
    all_text = text_conversion_ocr_capital(all_text)
    hasil = []

    for i, char1 in enumerate(sementara):
        char2 = all_text[i] if i < len(all_text) else ''
        hasil.append(char2 if not char1.isalnum() else char1)
        
    return ''.join(hasil)

def convert_chars_at_positions(string, char_map):
    for position, target_char, replacement_char in char_map:
        if position < len(string) and string[position] == target_char:
            string = string[:position] + replacement_char + string[position + 1:]
    return string

def deteksi_ocr(image):
    reader = easyocr.Reader(['id'])
    results = reader.readtext(image)
    array_text = [(text, prob) for (_, text, prob) in results]
    avg_prob = sum(prob for (_, prob) in array_text) / len(array_text) if array_text else 0
    print("average probability:", avg_prob)

    array_text.sort(key=lambda x: (not "MH" in x[0], x[0]))
    all_text = ' '.join(text for (text, _) in array_text[::-1]).replace(' ', '')
    return tampilkan_teks_fifo(all_text, image)

def tampilkan_teks_fifo(all_text, image):
    if re.search(r'[^a-zA-Z0-9]', all_text):
        text_free_symbol = karakter_tidak_valid(image, all_text)
        if re.search(r'[^a-zA-Z0-9]', text_free_symbol):
            print("OCR Tidak berhasil. gambar terlalu jelek")
            text_free_symbol = all_text
    else:
        text_free_symbol = all_text
    return text_conversion_ocr_capital(text_free_symbol)

def extract_information_from_plate(image_path):
    global metode_ocr
    # Load image using OpenCV
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['id'])
    results = reader.readtext(image)

    for (bbox, text, prob) in results:
        bbox = [(int(point[0]), int(point[1])) for point in bbox]
        cropped_image = image[bbox[0][1]:bbox[2][1], bbox[0][0]:bbox[2][0]]
        testPreprocess = preprocessing_image(cropped_image)
        return deteksi_ocr(testPreprocess)

def preprocessing_image(img):
    global metode_ocr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    cleaned_img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(5, 5))
    enhanced_img = clahe.apply(cleaned_img)
    metode_ocr["original"] = deteksi_ocr(enhanced_img)

    inverted_img = cv2.bitwise_not(enhanced_img)
    metode_ocr["inverted"] = deteksi_ocr(inverted_img)

    (T, thresh) = cv2.threshold(inverted_img, 77, 255, cv2.THRESH_BINARY)
    metode_ocr["negative"] = deteksi_ocr(thresh)

    return inverted_img

def fifo_compare_and_vote(ocr_results):
    max_length = max(len(ocr_results["original"]), len(ocr_results["negative"]), len(ocr_results["inverted"]))
    fifo_original = deque(ocr_results["original"].ljust(max_length))
    fifo_negative = deque(ocr_results["negative"].ljust(max_length))
    fifo_inverted = deque(ocr_results["inverted"].ljust(max_length))

    final_result = []

    for _ in range(max_length):
        char_original = fifo_original.popleft()
        char_negative = fifo_negative.popleft()
        char_inverted = fifo_inverted.popleft()

        if char_original == ' ':
            char_original = char_inverted
        if char_negative == ' ':
            char_negative = char_inverted

        votes = [char_original, char_negative, char_inverted]
        vote_counts = {char: votes.count(char) for char in set(votes)}
        max_vote = max(vote_counts.values())
        candidates = [char for char, count in vote_counts.items() if count == max_vote]

        final_char = candidates[0] if len(candidates) == 1 else char_inverted
        final_result.append(final_char)
    
    return ''.join(final_result).strip()

def process_image(image_path):
    global metode_ocr
    hasil_ocr = extract_information_from_plate(image_path)
    
    # Jalankan perbandingan FIFO dan voting
    hasil_akhir = fifo_compare_and_vote(metode_ocr)
    
    # Tampilkan hasil akhir setelah voting
    print("\nHasil Akhir Setelah Voting:")
    print(hasil_akhir)
    
    return hasil_akhir
