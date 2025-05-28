import json
import os
import time
import threading
from googletrans import Translator
from tqdm import tqdm
import concurrent.futures
import langdetect

# Đường dẫn file
INPUT_FILE = "output_merged_data.json"
OUTPUT_FILE = "output_merged_data_fixed.jsonl"  # File chính, append tất cả dữ liệu
UNTRANSLATED_FILE = "untranslated_items.jsonl"  # File lưu các phần tử chưa dịch
CHECKPOINT_FILE = "retranslation_checkpoint.json"
CHUNK_SIZE = 100

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

# Hàm kiểm tra xem văn bản có phải tiếng Việt không
def is_vietnamese(text):
    try:
        lang = langdetect.detect(text)
        return lang == 'vi'
    except:
        return False

# Hàm dịch một item với retry
def translate_item(item, translator, max_retries=3):
    needs_retranslation = not (is_vietnamese(item["question"]) and is_vietnamese(item["answer"]))
    if not needs_retranslation:
        return item, None  # Không cần dịch lại

    for attempt in range(max_retries):
        try:
            translated_question = translator.translate(item["question"], src='en', dest='vi').text if not is_vietnamese(item["question"]) else item["question"]
            translated_answer = translator.translate(item["answer"], src='en', dest='vi').text if not is_vietnamese(item["answer"]) else item["answer"]
            result = {
                "question": translated_question,
                "answer": translated_answer,
                "image_id": item["image_id"]
            }
            if is_vietnamese(translated_question) and is_vietnamese(translated_answer):
                return result, None
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return result, "Not fully translated to Vietnamese"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return item, e  # Giữ nguyên nếu không dịch được

# Hàm xử lý từng chunk với đa luồng
def process_chunk(chunk_data, chunk_idx, timing_data, output_file, untranslated_file):
    output_chunk = []
    untranslated_items = []
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(translate_item, item, Translator()): i for i, item in enumerate(chunk_data)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunk_data), desc=f"Processing chunk {chunk_idx}"):
            item_idx = futures[future]
            translated_item, error = future.result()
            output_chunk.append(translated_item)
            if error:
                print(f"Lỗi khi dịch lại ở chunk {chunk_idx}, item {item_idx}: {error}")
                untranslated_items.append(translated_item)  # Lưu item chưa dịch hoàn toàn

    output_chunk = [item for _, item in sorted(zip(futures.values(), output_chunk), key=lambda x: x[0])]
    
    # Append toàn bộ chunk vào file chính
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in output_chunk:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Append các phần tử chưa dịch hoàn toàn vào file riêng
    if untranslated_items:
        with open(untranslated_file, 'a', encoding='utf-8') as f:
            for item in untranslated_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    elapsed_time = time.time() - start_time
    timing_data.append(elapsed_time)
    print(f"Đã xử lý chunk {chunk_idx} trong {format_time(elapsed_time)}")
    print(f"Đã append {len(output_chunk)} phần tử vào {output_file}, {len(untranslated_items)} phần tử chưa dịch vào {untranslated_file}")
    return output_chunk, elapsed_time

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    total_chunks = (total_items + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Tổng số phần tử: {total_items}")
    print(f"Tổng số chunk: {total_chunks}")
    
    checkpoint = load_checkpoint()
    last_processed_chunk = checkpoint["last_processed_chunk"]
    remaining_chunks = total_chunks - (last_processed_chunk + 1)
    print(f"Chunk đã xử lý: {last_processed_chunk + 1}, còn lại: {remaining_chunks}")
    
    timing_data = []

    # Xóa file output cũ nếu tồn tại để bắt đầu ghi mới
    for file_path in [OUTPUT_FILE, UNTRANSLATED_FILE]:
        if os.path.exists(file_path):
            os.remove(file_path)

    for chunk_idx, i in enumerate(range(0, total_items, CHUNK_SIZE)):
        if chunk_idx <= last_processed_chunk:
            continue
        
        chunk_data = data[i:i + CHUNK_SIZE]
        translated_chunk, elapsed_time = process_chunk(chunk_data, chunk_idx, timing_data, OUTPUT_FILE, UNTRANSLATED_FILE)
        
        with checkpoint_lock:
            save_checkpoint(chunk_idx)
        
        if timing_data:
            avg_time_per_chunk = sum(timing_data) / len(timing_data)
            chunks_left = total_chunks - (chunk_idx + 1)
            estimated_time_left = avg_time_per_chunk * chunks_left
            print(f"Thời gian trung bình mỗi chunk: {format_time(avg_time_per_chunk)}")
            print(f"Ước lượng thời gian còn lại: {format_time(estimated_time_left)} ({chunks_left} chunks)")
            print("-" * 50)

    print(f"Đã append tất cả dữ liệu vào {OUTPUT_FILE}. Các phần tử chưa dịch hoàn toàn được lưu vào {UNTRANSLATED_FILE}.")

if __name__ == "__main__":
    main()