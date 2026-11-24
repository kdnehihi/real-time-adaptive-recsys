# Tổng quan dự án Real-Time Adaptive Recommender System

Tài liệu này giải thích dự án theo cách dễ đọc, từ dữ liệu KuaiRand ban đầu đến
baseline, Two-Tower, adaptive retrieval và online replay. Đây là tài liệu tổng
quan; các tài liệu kỹ thuật chi tiết hơn vẫn nằm trong cùng thư mục `docs/`.

## 1. Dự án đang giải quyết bài toán gì?

Mục tiêu là xây dựng hệ thống gợi ý video ngắn có hai loại hiểu biết về user:

- **Long-term preference:** sở thích tương đối ổn định, được tổng hợp từ lịch sử
  dài hơn của user.
- **Short-term preference:** ý định gần đây, ví dụ user vừa xem liên tiếp một
  nhóm video hoặc tác giả cụ thể.

Luồng production mong muốn là:

```text
interaction mới
-> cập nhật online user state và recent sequence
-> V4 retrieval lấy khoảng 1.000 candidates
-> ranker sắp xếp lại candidates
-> Top-K recommendation
```

Hiện tại dự án đã có nền tảng dữ liệu, các baseline, adaptive retrieval và mô
phỏng online state. Kafka, ANN serving thực tế và recommendation API chưa được
triển khai.

## 2. Các công cụ đã sử dụng

| Công cụ | Vai trò trong dự án |
| --- | --- |
| Python 3.11 | Ngôn ngữ chính cho preprocessing, model và evaluation |
| Java 17 + PySpark 3.5 | Xử lý CSV/Parquet, join lớn, window, session và Gold build |
| Spark SQL | Tính point-in-time aggregate và kiểm tra chất lượng dữ liệu |
| Parquet | Định dạng lưu trữ chính sau Raw; nhanh và tiết kiệm hơn CSV |
| PyArrow | Đọc Parquet theo shard trong các pipeline Colab/evaluation |
| DuckDB | Chuẩn bị một số dataset corrected-V4 theo kiểu out-of-core |
| PyTorch | Feature embedding, MLP Two-Tower, Transformer sequence encoder |
| Spark MLlib ALS | Collaborative-filtering baseline |
| MLflow | Track local smoke/baseline experiment và metric |
| Optuna | Có hỗ trợ tuning interaction-strength/weight ở một số pipeline |
| Google Colab | Train/evaluate các model lớn bằng GPU và High-RAM |
| Google Drive | Lưu Gold data và một số manifest/report nhỏ |
| Hugging Face private model repo | Lưu checkpoint, runtime ZIP, corrected delta và test artifacts lớn |
| Git/GitHub | Version code, notebook, config, test và documentation; không lưu data lớn |
| pytest | Unit/smoke/regression test cho data contract, encoder và model |

Nguyên tắc lựa chọn công cụ là dùng Spark cho dữ liệu lớn, PyTorch cho model,
và tránh đưa toàn bộ dataset vào pandas nếu không cần thiết.

## 3. Thiết kế storage và data layers

Project dùng cấu trúc lakehouse tối giản:

```text
data/raw       CSV gốc, bất biến
data/bronze    CSV đã chuyển sang Parquet, thay đổi tối thiểu
data/silver    dữ liệu sạch, đúng kiểu và còn tương đối đầy đủ
data/gold      dữ liệu được thiết kế cho từng task/model
data/features  vị trí dành cho offline/online feature tables sau này
```

### Raw

Raw chứa các CSV KuaiRand. Thư mục data được `.gitignore`; GitHub chỉ giữ code
để tái tạo dữ liệu.

### Bronze

Bronze chuyển từng CSV sang Parquet nhưng chưa áp nhiều business logic. Mục
đích là đọc nhanh hơn, giữ schema ổn định hơn và không phải parse CSV ở mọi lần
EDA/build.

### Silver

Silver gồm bốn nhóm chính:

- `interactions`: event của user với video, timestamp, watch ratio và các tín
  hiệu click/like/follow/hate/long-view.
- `users`: user profile và các anonymous `onehot_feat*`; đây là các feature đã
  được KuaiRand mã hóa/ẩn danh, không phải one-hot vector do project tự sinh.
- `videos_basic`: metadata như author, video type, upload type, duration,
  resolution và upload date.
- `videos_statistics`: counter/rate tổng hợp của video; được giữ đầy đủ ở
  Silver nhưng không mặc định đưa vào model nếu không chứng minh được snapshot
  time, vì có nguy cơ future leakage.

Silver ưu tiên đầy đủ và sạch. Feature selection thuộc về Gold/model layer.

### Gold

Gold được tách theo mục đích:

```text
data/gold/als/         collaborative-filtering baseline
data/gold/two_tower/   static/point-in-time Two-Tower
data/gold/adaptive/    recent sequence, serving query và online replay
```

Một row Two-Tower/Adaptive Gold có thể hiểu đơn giản là:

```text
trạng thái user trước thời điểm t
+ feature của video tại thời điểm t
+ context tại thời điểm t
-> phản ứng quan sát được của user tại t (label)
```

## 4. Temporal split: Train, Validation và Test

Dữ liệu được chia theo thời gian, không random theo row:

| Split | Khoảng thời gian |
| --- | --- |
| Train | Trước `2022-04-30 07:30:11` |
| Validation | Từ cutoff Train đến trước `2022-05-04 08:45:20` |
| Test | Từ cutoff Validation trở đi |

Lý do:

- Train mô phỏng dữ liệu hệ thống đã biết.
- Validation dùng để chọn hyperparameter, epoch và weight.
- Test chỉ được dùng một lần cho kết luận cuối, không được dùng để chỉnh model.

Các quy tắc chống leakage:

1. Feature của example tại `t` chỉ dùng event có thời gian `< t`.
2. Vocabulary categorical và ID mapping được fit trên Train.
3. Mean/std, p01/p99 và imputation stats được fit trên Train.
4. Validation/Test chỉ transform bằng artifact của Train.
5. Outcome của current event như `long_view`, `like`, `watch_ratio` là label,
   không phải input.
6. Candidate chỉ được recommend nếu đã tồn tại tại query time.
7. Item đã xem trước đó được lọc, ngoại trừ trường hợp exact target cố ý là một
   repeated item.

## 5. Quy mô Two-Tower Gold hiện tại

| Split/table | Số row |
| --- | ---: |
| Train | 8.219.912 |
| Validation | 1.767.868 |
| Test | 1.768.293 |
| Full event history | 11.756.073 |
| Static item metadata | 4.371.868 |

Target distribution:

| Split | Strong positive | Observed negative | Ambiguous/weak |
| --- | ---: | ---: | ---: |
| Train | 2.234.745 | 551.981 | 5.433.186 |
| Validation | 477.290 | 118.510 | 1.172.068 |
| Test | 468.278 | 114.793 | 1.185.222 |

Định nghĩa label:

- `STRONG_POSITIVE`: long-view, like, comment, forward hoặc follow.
- `OBSERVED_NEGATIVE`: không positive và có hate, hoặc click nhưng watch ratio
  rất thấp.
- `AMBIGUOUS_WEAK`: có impression/event nhưng chưa đủ bằng chứng positive hay
  negative.
- Cặp user-video chưa quan sát **không tự động được coi là negative**.

Cold item là thách thức lớn: khoảng 64,6% validation và 77,6% test event có
video chưa xuất hiện trong Train interactions. Vì vậy model chỉ dựa vào
`video_id` sẽ không đủ; metadata/content branch rất quan trọng.

## 6. Feature design

### User-side input

User Tower hiện dùng các nhóm:

- ID/profile: `user_id`, `user_active_degree`, các anonymous user features.
- Long-term aggregates: số event trước đó, like/long-view/follow/hate rate,
  watch ratio và play-time trung bình.
- Session state: event index, elapsed time và session-level behavior rates.
- Drift features: chênh lệch session preference với long-term preference.
- Context: hour, day-of-week, tab và random/standard exposure flag.

### Item-side input

- `video_id`, `author_id` khi branch tương ứng được dùng.
- `video_type`, `upload_type`.
- duration, aspect ratio, visible status, music type và upload age.
- item popularity/engagement aggregates được tính point-in-time.

### Recent sequence

Adaptive Gold lưu tối đa 30 event gần nhất toàn cục và 10 event trong session:

```text
recent_video_ids
recent_target_classes
recent_engagement_strengths
recent_long_views / likes / hates / clicks
recent_time_ms
```

Các array phải thẳng hàng và mọi timestamp phải nhỏ hơn query time.

### Encoding

- Index `0` dành cho OOV/unknown.
- Categorical/ID được đưa qua trainable embedding tables.
- Numeric được impute, clip p01/p99 và standardize bằng Train stats.
- Unseen video dùng OOV `video_id`, nhưng vẫn có thể tạo embedding từ metadata.

## 7. Các model đã xây dựng

### Popularity baseline

Recommend các item phổ biến nhất. Đây là lower baseline đơn giản và là sanity
check cho evaluation.

### Spark ALS baseline

ALS học latent factor từ ma trận user-item implicit feedback. Event strength:

```text
0.5 * clipped_watch_ratio
+ 1.0 * long_view
+ 1.5 * like/comment/forward
+ 2.0 * follow
```

Các event lặp của cùng user-item được cộng rồi dùng `log1p`. Hyperparameter đã
chọn: rank 64, regularization 0.05, alpha 10, 8 iterations.

Kết quả test lịch sử:

| Model | NDCG@10 | Recall@10 | HitRate@10 |
| --- | ---: | ---: | ---: |
| Popularity | 0,00290 | 0,00442 | 0,04317 |
| ALS | 0,01152 | 0,01786 | 0,16327 |

ALS vượt popularity, nhưng không xử lý tốt cold item và không học short-term
sequence.

### Static Two-Tower V1–V3

Hai tower độc lập:

```text
user features -> User MLP -> user embedding
item features -> Item MLP -> item embedding
score = dot(user embedding, item embedding)
```

- V1: basic in-batch negatives.
- V2: full train item vocabulary và sampled negatives.
- V3: mixed in-batch + random + popularity-weighted negatives; là static neural
  baseline mạnh nhất của nhánh ban đầu.

Item embeddings có thể precompute; retrieval production sau này có thể dùng
ANN mà không thay kiến trúc model.

### V4 Adaptive Transformer Retrieval

V4 giữ Two-Tower nhưng thay User Tower bằng:

```text
long-term/latest user state -----------┐
                                      + learned gate -> user embedding
recent events -> Transformer encoder --┘
```

Mỗi sequence token kết hợp video embedding, event type, engagement signals và
time/recency. Self-attention học event nào trong recent history quan trọng.
Item Tower vẫn độc lập để có thể precompute và retrieve bằng matrix search/ANN.

### V4.1 và V4.2 author branch

- V4.1 học một author embedding branch lớn. Kết quả không cải thiện ổn định và
  bị loại.
- V4.2 dùng exact-author affinity đơn giản, có recency weighting, kết hợp
  popularity bias. Trên legacy validation holdout, cấu hình được chọn là
  `author_weight=0.2`, `popularity_weight=0.02`, `recency_tau=3.0` và cải thiện
  HR@1000 có confidence interval dương.

V4.2 có thể triển khai production bằng cách union candidates từ V4 ANN và các
video của recent authors, rồi cộng score bias.

### V5 ranker

V5 được thiết kế để rerank cùng một candidate set của V4 và nhận thêm V4 score,
user state, sequence representation, item/context features.

Phiên bản V5 đã thử bị **reject**: chỉ khoảng 1,22% train target xuất hiện tự
nhiên trong V4 Top-128; phần lớn positive phải bị ép vào candidate group, gây
train-serving mismatch. V5 làm giảm NDCG/HitRate so với thứ tự V4. Không nên
tiếp tục tối ưu ranker trước khi retrieval coverage tốt hơn.

### V6 online replay

V6 không phải model mới. Nó nạp V4/V4.2 và replay theo timestamp:

```text
events trước t -> update bounded user/session state
query tại t -> đọc state trước t
-> retrieval/reranking
-> sau đó mới append toàn bộ events tại t
```

Reducer giữ 30 global events và 10 session events. Replay đã phát hiện legacy
Gold sequence có lỗi no-backfill ở same-timestamp batch. Đây là lý do project
đã xây corrected sequence contract và retrain Corrected V4.

## 8. Legacy sequence và corrected sequence

Legacy builder từng lấy N row trước, sau đó loại event cùng timestamp với query.
Khi event bị loại, builder không backfill event cũ hơn. Kết quả vẫn không dùng
future event nhưng sequence ngắn hơn và phụ thuộc tie ordering.

Corrected contract làm đúng thứ tự:

```text
group events theo timestamp
-> loại toàn bộ current timestamp batch
-> sau đó mới lấy 30 global / 10 session events gần nhất
```

Independent Spark builder và online reducer đã đạt exact parity trên toàn bộ 18
sequence fields của 7.655 validation queries. Corrected delta train/validation
đã được materialize và Corrected V4 đã được retrain.

Trạng thái so sánh:

- Corrected retraining tốt hơn legacy checkpoint khi cả hai cùng nhận corrected
  input, nhưng HR@1000 confidence interval vẫn cắt zero.
- Thêm fixed author affinity giúp Corrected V4 tốt hơn matched Corrected V4 và
  phục hồi cold-item metric.
- So với Legacy V4.2, point estimate tốt hơn nhưng interval vẫn cắt zero; vì
  vậy chưa đủ bằng chứng để gọi là serving winner.

## 9. Evaluation được thiết kế như thế nào?

### Vì sao không chỉ nhìn training loss?

Loss cho biết optimizer đang fit objective, không đảm bảo model retrieve đúng
item trong catalog hàng trăm nghìn/millions video. Model được chọn bằng ranking
metric trên protocol cố định.

### Metric chính

- `HitRate@K`: trong Top-K có ít nhất một relevant item hay không.
- `Recall@K`: bao nhiêu relevant items được lấy lại trong Top-K.
- `NDCG@K`: relevant item có được đặt gần đầu danh sách hay không.
- `MRR`: reciprocal rank của relevant item đầu tiên.
- Cold/warm segment metrics: tách item đã thấy và chưa thấy trong Train.
- Clustered bootstrap by user: tạo confidence interval mà không coi nhiều query
  của cùng user là các quan sát độc lập.

### Corrected A2 validation protocol

- Tối đa 12 deterministic point-in-time queries mỗi user.
- 11.758 validation queries, 984 users.
- 854.893 time-eligible candidate items.
- Exact current positive là primary target.
- Các model dùng cùng candidate set, eligibility mask và seen-item filter.
- Có segment theo history length, warm/cold và timeline.

V4 adaptive cải thiện exact HR@1000 từ 1,482% của static ablation lên 2,356%,
với paired CI95 hoàn toàn dương. Điều này chứng minh recent sequence mang thêm
tín hiệu có ích. Tuy nhiên cold retrieval vẫn là bottleneck lớn.

### Locked test

Test vật lý đã được rebuild độc lập:

- 1.768.293 examples, 986 users và 886.302 items.
- Không thiếu user/item join và không có timestamp violation.
- Locked protocol có 11.737 exact-target queries từ 983 users.
- Khoảng 74,1% target là cold so với Train.

Notebook 33 đóng băng weight từ Validation và so sánh Corrected V4, Corrected
V4 + author affinity và Legacy V4.2. Corrected test sequence đạt đủ 11.737 row,
không có timestamp/alignment violation và resolve millisecond event time bằng
đúng Gold key:

```text
example_id = SHA256("example||event_id||time_ms")
```

Fixed author affinity tăng Corrected HR@1000 từ 1,2394% lên 1,4513% và cải thiện
cold HR@1000 rõ rệt. Legacy V4.2 đạt 1,4174%; Corrected cao hơn nhẹ nhưng paired
CI95 của chênh lệch vẫn cắt zero. Warm HR@1000 của Corrected thấp hơn Legacy
(4,0184% so với 4,4861%), nên promotion gate cuối không pass. Không được thay
weight dựa trên kết quả Test này.

## 10. Artifact được lưu ở đâu?

### GitHub

Lưu code, notebook, config, test và docs. Không lưu raw/processed data hoặc model
checkpoint lớn.

### Google Drive

Các root chính trên Colab:

```text
/content/drive/MyDrive/recsys/data/gold/two_tower/v1_colab/
/content/drive/MyDrive/recsys/data/gold/adaptive/v1/
```

Drive nên ưu tiên Gold gốc và report nhỏ, không lưu mọi intermediate.

### Hugging Face private repository

```text
khoa1505/recsys-artifacts
```

Lưu checkpoint lớn, runtime ZIP, corrected sequence delta và adaptive test
artifact. Colab dùng `HF_TOKEN` để đọc và `HF_WRITE_TOKEN` để upload.

## 11. Trạng thái hiện tại và bước tiếp theo

Đã hoàn thành:

- Raw CSV -> Bronze Parquet -> Silver clean tables.
- ALS Gold và ALS/Popularity baselines.
- Point-in-time Two-Tower Gold và feature encoders.
- Static Two-Tower V1–V3.
- V4 Transformer adaptive retrieval.
- V5 experiment và quyết định reject.
- V4.1/V4.2 author experiments.
- Corrected sequence contract, corrected delta và Corrected V4 retraining.
- V6 timestamp replay và true online-state reducer.
- Independent test rebuild, locked test protocol và final Notebook 33 report.
- V4.3 static-retention validation; branch bị reject và không mở lại locked Test.

Việc ngay trước mắt:

1. Chạy Notebook 35 để benchmark FAISS HNSW so với exact full-policy của
   Corrected V4.2 trên Validation holdout.
2. Chọn ANN config chỉ khi Top-1000 overlap, target HitRate và output coverage
   đều qua gate; index lớn chỉ nằm trên local Colab disk.
3. Không train/tune thêm hoặc chạy lại locked Test ở bước serving benchmark.

Sau final test, hướng cải thiện có giá trị nhất là:

1. Thêm semantic item features: tag/text/audio/visual embeddings để giải quyết
   cold video và unseen author.
2. Dùng multi-interest user representation thay vì ép mọi recent preference
   vào một vector.
3. Tăng candidate coverage bằng multi-source retrieval: ANN Two-Tower + recent
   author + popularity/freshness/content sources.
4. Chỉ train V5 ranker sau khi positive natural coverage trong retrieved
   candidate set đủ tốt.
5. Sau Notebook 35, đóng gói ANN config đã pass thành retrieval service và đo
   request-level latency/concurrency.
6. Materialize incremental long-term state và nối reducer với Kafka/feature
   store/serving API.

## 12. Thứ tự đọc code/notebook khuyến nghị

Nếu muốn hiểu dự án mà không chạy lại toàn bộ:

1. `01_kuairand_data_understanding.ipynb`: hiểu từng source table.
2. `docs/gold_data_design.md`: hiểu ALS và Two-Tower Gold.
3. `06_two_tower_feature_encoding_walkthrough.ipynb`: raw feature thành tensor.
4. `09_colab_two_tower_v3_final_training.ipynb`: static Two-Tower baseline.
5. `11_colab_adaptive_gold_builder.ipynb`: recent sequence data.
6. `docs/adaptive_evaluation.md`: lịch sử quyết định V4/V5/V6 và corrected data.
7. `28` -> `31`: sequence correction, retraining và author calibration.
8. `32` -> `34`: independent Test, final evaluation và validation-only
   warm-retention rejection.
9. `35`: ANN serving parity, latency và memory benchmark.

Tài liệu chi tiết liên quan:

- `docs/storage_layer.md`
- `docs/gold_data_design.md`
- `docs/two_tower_feature_encoding.md`
- `docs/two_tower_training.md`
- `docs/adaptive_data_design.md`
- `docs/adaptive_evaluation.md`
