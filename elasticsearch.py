import os
import glob
import time
import pickle
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# -----------------------------------------------------------------------------
# 1. Đường dẫn tệp và Cấu hình Elasticsearch
# -----------------------------------------------------------------------------
METADATA_DIR = "/home/sysadmin/AIC_2026/Keyframe/metadata"
PKL_ASR_OCR = "/home/sysadmin/AIC_2026/OCR/_plus_asr/asr_ocr_merged.pkl"

OUTPUT_DIR = "/home/sysadmin/AIC_2026/Keyframe/metadata"
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "metadata_merged_with_asr_ocr.pkl")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metadata_merged_with_asr_ocr.csv")

TARGET_INDEX = "video_metadata_merged_v1"
ALIAS_NAME = "asr_ocr_merged"

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "1ND2kR_8MUQL_7t4G+kD"

def main():
    start_all = time.time()
    print("=== BẮT ĐẦU QUÁ TRÌNH KẾT HỢP METADATA VỚI ASR & OCR ===")

    # 1. Kiểm tra kết nối Elasticsearch
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        request_timeout=120,
    )
    if not es.ping():
        raise ConnectionError(f"Không thể kết nối tới Elasticsearch tại {ES_HOST}")
    print(f"-> Đã kết nối Elasticsearch phiên bản: {es.info()['version']['number']}")

    # 2. Kiểm tra xem file gộp đã có sẵn chưa
    if os.path.exists(OUTPUT_PKL):
        print(f"-> Đã tìm thấy tệp gộp sẵn tại {OUTPUT_PKL}. Đang nạp...")
        with open(OUTPUT_PKL, "rb") as f:
            df_merged = pickle.load(f)
        print(f"-> Đã nạp {len(df_merged):,} dòng từ file gộp.")
    else:
        # Nạp và gộp 16 shard CSV metadata
        shard_files = sorted(glob.glob(os.path.join(METADATA_DIR, "metadata_shard_*.csv")))
        print(f"-> Tìm thấy {len(shard_files)} file shard CSV metadata.")
        
        df_meta_list = []
        for sf in shard_files:
            df_shard = pd.read_csv(sf)
            df_meta_list.append(df_shard)
        
        df_meta = pd.concat(df_meta_list, ignore_index=True)
        print(f"-> Đã tải tổng cộng {len(df_meta):,} dòng từ metadata shards.")

        # Chuẩn hóa khóa
        df_meta["video_name"] = df_meta["video_name"].astype(str).str.strip()
        df_meta["frame_id"] = pd.to_numeric(df_meta["frame_id"], errors="coerce").fillna(0).astype(int)
        
        # Khử trùng lặp trong metadata
        len_before = len(df_meta)
        df_meta = df_meta.drop_duplicates(subset=["video_name", "frame_id"], keep="first")
        print(f"-> Đã khử trùng lặp metadata: còn {len(df_meta):,} unique frames (loại bỏ {len_before - len(df_meta)} trùng lặp).")

        # Nạp dữ liệu ASR + OCR từ file pkl hiện có
        print(f"-> Đang nạp dữ liệu ASR + OCR từ: {PKL_ASR_OCR} ...")
        with open(PKL_ASR_OCR, "rb") as f:
            df_asr_ocr = pickle.load(f)
        print(f"-> Đã tải {len(df_asr_ocr):,} dòng từ file pickle ASR+OCR.")

        # Chuẩn hóa khóa df_asr_ocr
        df_asr_ocr["video_id"] = df_asr_ocr["video_id"].astype(str).str.strip()
        df_asr_ocr["frame_id_ocr"] = pd.to_numeric(df_asr_ocr["frame_id_ocr"], errors="coerce").fillna(0).astype(int)
        df_asr_ocr = df_asr_ocr.drop_duplicates(subset=["video_id", "frame_id_ocr"], keep="first")

        # Gộp dữ liệu
        print("-> Đang gộp dữ liệu (Join) theo (video_name == video_id) và (frame_id == frame_id_ocr)...")
        df_merged = pd.merge(
            df_meta,
            df_asr_ocr,
            how="left",
            left_on=["video_name", "frame_id"],
            right_on=["video_id", "frame_id_ocr"],
        )

        df_merged["video_id"] = df_merged["video_name"]
        df_merged["frame_id_ocr"] = df_merged["frame_id"]
        
        df_merged["title"] = df_merged["title"].fillna("").astype(str)
        df_merged["video_description"] = df_merged["video_description"].fillna("").astype(str)
        df_merged["frame_description"] = df_merged["frame_description"].fillna("").astype(str)
        df_merged["frame_name"] = df_merged["frame_name"].fillna("").astype(str)
        df_merged["scene_id"] = pd.to_numeric(df_merged["scene_id"], errors="coerce").fillna(0).astype(int)
        df_merged["ocr_text"] = df_merged["ocr_text"].fillna("").astype(str)
        df_merged["clean_text"] = df_merged["clean_text"].fillna("").astype(str)

        print(f"-> Đang lưu kết quả ra file pickle: {OUTPUT_PKL} ...")
        with open(OUTPUT_PKL, "wb") as f:
            pickle.dump(df_merged, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"-> Đang lưu bản CSV: {OUTPUT_CSV} ...")
        csv_cols = [c for c in df_merged.columns if c != "bert_embedding"]
        df_merged[csv_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"-> Dữ liệu tổng hợp: {len(df_merged):,} dòng.")
    matched_count = (df_merged["ocr_text"] != "").sum()
    print(f"   + Số frame có OCR text: {matched_count:,}")
    asr_count = (df_merged["clean_text"] != "").sum()
    print(f"   + Số frame có ASR text: {asr_count:,}")

    # 3. Thiết lập Mapping và tạo Index mới trên Elasticsearch
    if es.indices.exists(index=TARGET_INDEX):
        print(f"-> Index {TARGET_INDEX} đã tồn tại, đang tiến hành xóa để tạo mới...")
        es.indices.delete(index=TARGET_INDEX)

    index_mapping = {
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "video_name": {"type": "keyword"},
                "frame_id": {"type": "integer"},
                "frame_id_ocr": {"type": "integer"},
                "scene_id": {"type": "integer"},
                "frame_name": {"type": "keyword"},
                "title": {"type": "text"},
                "video_description": {"type": "text"},
                "frame_description": {"type": "text"},
                "ocr_text": {"type": "text"},
                "clean_text": {"type": "text"},
                "bert_embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1"
            }
        }
    }

    print(f"-> Đang tạo index '{TARGET_INDEX}' trên Elasticsearch...")
    es.indices.create(index=TARGET_INDEX, body=index_mapping)

    # 4. Bulk Upload lên Elasticsearch
    print(f"-> Đang chuẩn bị và upload {len(df_merged):,} documents...")
    def generate_actions():
        for _, row in df_merged.iterrows():
            doc_source = {
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "frame_id": int(row["frame_id"]),
                "frame_id_ocr": int(row["frame_id_ocr"]),
                "scene_id": int(row["scene_id"]),
                "frame_name": str(row["frame_name"]),
                "title": row["title"],
                "video_description": row["video_description"],
                "frame_description": row["frame_description"],
                "ocr_text": row["ocr_text"],
                "clean_text": row["clean_text"],
            }
            
            # Chỉ gán bert_embedding khi có vector hợp lệ (tránh lỗi cosine similarity với zero magnitude)
            emb = row.get("bert_embedding")
            if isinstance(emb, (np.ndarray, list)):
                vector = emb.tolist() if isinstance(emb, np.ndarray) else emb
                if len(vector) == 768:
                    doc_source["bert_embedding"] = vector

            doc_id = f"{row['video_id']}_{row['frame_id']}"
            yield {
                "_index": TARGET_INDEX,
                "_id": doc_id,
                "_source": doc_source
            }

    upload_start = time.time()
    success_count, failed = bulk(
        es,
        generate_actions(),
        chunk_size=2000,
        request_timeout=180,
        stats_only=True
    )
    upload_elapsed = time.time() - upload_start
    print(f"-> Upload hoàn tất! Đã thêm thành công {success_count:,} documents (thời gian: {upload_elapsed:.1f}s).")

    # Bật lại refresh_interval
    es.indices.put_settings(index=TARGET_INDEX, body={"index": {"refresh_interval": "1s"}})
    es.indices.refresh(index=TARGET_INDEX)

    # 5. Cập nhật Alias 'asr_ocr_merged' trỏ tới index mới
    print(f"-> Đang cấu hình alias '{ALIAS_NAME}' trỏ sang '{TARGET_INDEX}'...")
    
    # Nếu index cũ có tên là asr_ocr_merged (không phải alias), ta tiến hành xóa để nhường tên cho Alias
    if es.indices.exists(index=ALIAS_NAME) and not es.indices.exists_alias(name=ALIAS_NAME):
        print(f"   + Index cũ '{ALIAS_NAME}' là một index vật lý. Đang tiến hành xóa...")
        es.indices.delete(index=ALIAS_NAME)

    # Nếu alias đã tồn tại trỏ vào index khác, xóa alias cũ
    if es.indices.exists_alias(name=ALIAS_NAME):
        existing_indices = list(es.indices.get_alias(name=ALIAS_NAME).keys())
        for idx in existing_indices:
            es.indices.delete_alias(index=idx, name=ALIAS_NAME)

    es.indices.put_alias(index=TARGET_INDEX, name=ALIAS_NAME)
    print(f"-> Alias '{ALIAS_NAME}' hiện đã trỏ thành công tới '{TARGET_INDEX}'.")

    # 6. Kiểm tra xác nhận
    count_res = es.count(index=ALIAS_NAME)
    print(f"-> XÁC NHẬN: Số document trong alias '{ALIAS_NAME}': {count_res['count']:,}")

    total_time = time.time() - start_all
    print(f"=== TẤT CẢ HOÀN TẤT THÀNH CÔNG TRONG {total_time:.1f} GIÂY ===")

if __name__ == "__main__":
    main()
