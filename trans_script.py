import json
import os
import time
import threading
from googletrans import Translator
from tqdm import tqdm
import concurrent.futures

# Đường dẫn file
INPUT_FILE = "merged_data.json"
OUTPUT_DIR = "translated_chunks"
CHECKPOINT_FILE = "translation_checkpoint.json"

# Kích thước chunk
CHUNK_SIZE = 100  # Kích thước mỗi chunk

# Lock để đảm bảo an toàn khi cập nhật checkpoint
checkpoint_lock = threading.Lock()

# Hàm kiểm tra và tải checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"last_processed_chunk": -1}

# Hàm lưu checkpoint
def save_checkpoint(chunk_idx):
    checkpoint = {"last_processed_chunk": chunk_idx}
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=4)

# Hàm định dạng thời gian
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

# Hàm dịch một item với retry
def translate_item(item, translator, max_retries=3):
    for attempt in range(max_retries):
        try:
            translated_question = translator.translate(item["question"], src='en', dest='vi').text
            translated_answer = translator.translate(item["answer"], src='en', dest='vi').text
            return {
                "question": translated_question,
                "answer": translated_answer,
                "image_id": item["image_id"]
            }, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return {
                "question": item["question"],
                "answer": item["answer"],
                "image_id": item["image_id"]
            }, e

# Hàm xử lý từng chunk với đa luồng
def process_chunk(chunk_data, chunk_idx, timing_data):
    output_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx}.json")
    if os.path.exists(output_file):
        print(f"Chunk {chunk_idx} đã được xử lý trước đó, bỏ qua.")
        return 0  # Trả về 0 giây nếu bỏ qua

    start_time = time.time()
    translated_chunk = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(translate_item, item, Translator()): i for i, item in enumerate(chunk_data)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunk_data), desc=f"Processing chunk {chunk_idx}"):
            item_idx = futures[future]
            translated_item, error = future.result()
            translated_chunk.append(translated_item)
            
            if error:
                print(f"Lỗi khi dịch ở chunk {chunk_idx}, item {item_idx}: {error}")
    
    translated_chunk = [item for _, item in sorted(zip(futures.values(), translated_chunk), key=lambda x: x[0])]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_chunk, f, ensure_ascii=False, indent=4)
    
    with checkpoint_lock:
        save_checkpoint(chunk_idx)
    
    elapsed_time = time.time() - start_time
    timing_data.append(elapsed_time)
    print(f"Đã xử lý và lưu chunk {chunk_idx} trong {format_time(elapsed_time)}")
    return elapsed_time

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    total_chunks = (total_items + CHUNK_SIZE - 1) // CHUNK_SIZE  # Tổng số chunk
    print(f"Tổng số phần tử: {total_items}")
    print(f"Tổng số chunk: {total_chunks}")
    
    checkpoint = load_checkpoint()
    last_processed_chunk = checkpoint["last_processed_chunk"]
    remaining_chunks = total_chunks - (last_processed_chunk + 1)
    print(f"Chunk đã xử lý: {last_processed_chunk + 1}, còn lại: {remaining_chunks}")
    
    # Danh sách lưu thời gian xử lý từng chunk
    timing_data = []
    
    for chunk_idx, i in enumerate(range(0, total_items, CHUNK_SIZE)):
        if chunk_idx <= last_processed_chunk:
            continue
        chunk_data = data[i:i + CHUNK_SIZE]
        process_chunk(chunk_data, chunk_idx, timing_data)
        
        # Tính thời gian trung bình và dự đoán thời gian còn lại
        if timing_data:
            avg_time_per_chunk = sum(timing_data) / len(timing_data)
            chunks_left = total_chunks - (chunk_idx + 1)
            estimated_time_left = avg_time_per_chunk * chunks_left
            print(f"Thời gian trung bình mỗi chunk: {format_time(avg_time_per_chunk)}")
            print(f"Ước lượng thời gian còn lại: {format_time(estimated_time_left)} ({chunks_left} chunks)")
            print("-" * 50)

if __name__ == "__main__":
    main()