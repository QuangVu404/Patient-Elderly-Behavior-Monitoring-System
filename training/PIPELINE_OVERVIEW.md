# Tính năng và luồng huấn luyện trên Kaggle

## Phạm vi

Bộ staged được đánh số từ `00` đến `07`, nên thực tế có **8 notebook**: một notebook chuẩn bị
manifest, một notebook fine-tune pose, năm notebook tạo cache cho năm nguồn và một notebook
train classifier. `all_in_one_demo_kaggle.ipynb` là đường chạy demo thay thế, không chạy cùng
bộ staged khi huấn luyện chính thức.

Pipeline tạo hệ thống nhận biết `fall`/`non-fall` từ video theo luồng:

```text
5 Kaggle Inputs
    -> manifest và group split
    -> YOLO-Pose lấy 17 keypoint
    -> vector đặc trưng 89 chiều theo thời gian
    -> cửa sổ temporal
    -> MultiScaleTemporalGRU tự huấn luyện
    -> checkpoint, metric và model triển khai
```

## Tính năng chính

- Nhận năm family: FallVision mirror, CAUCAFall, URFD, MCFD và UCF101.
- Đọc cả video và chuỗi ảnh; CAUCAFall/URFD không cần ghép lại thành video.
- Kiểm tra mount, inventory canonical và provenance trước khi xử lý.
- Chia dữ liệu theo subject, sequence, scenario hoặc group; không chia ngẫu nhiên từng frame.
- Giữ tám camera của cùng MCFD scenario trong một split để tránh leakage.
- UCF101 chỉ cung cấp hard-negative và được deduplicate bằng filename canonical.
- Bỏ qua frame MPEG-4 lỗi có giới hạn thay vì làm dừng toàn bộ pipeline.
- Dùng SHA-256 để bảo đảm mọi cache được tạo từ cùng pose checkpoint.
- Classifier `MultiScaleTemporalGRU` được huấn luyện từ đầu, không dùng classifier tải sẵn.
- Cân bằng sampling khi train và chọn threshold bằng validation set.
- Có thể xuất classifier sang ONNX và OpenVINO.

## Mục đích tổng quát của từng notebook

| Notebook | Mục đích trong pipeline |
|---|---|
| `00_build_manifest_kaggle.ipynb` | **Lập danh mục và chia dữ liệu.** Biến năm nguồn có cấu trúc khác nhau thành một bảng thống nhất, rồi quyết định dữ liệu nào thuộc train, validation và test. Đây là nền móng cố định để các notebook sau dùng cùng một cách hiểu về dữ liệu. |
| `01_finetune_pose_kaggle.ipynb` | **Điều chỉnh bộ nhìn cơ thể người cho miền camera giám sát/ngã.** Bắt đầu từ YOLO-Pose có sẵn, tạo hoặc kết hợp nhãn keypoint và fine-tune một pose checkpoint dùng chung. Mục tiêu là lấy tọa độ cơ thể ổn định hơn trước khi học hành vi. |
| `02_cache_fallvision_kaggle.ipynb` | **Chuyển FallVision thành đặc trưng nhẹ.** Chạy pose model trên video fall/non-fall và lưu chuỗi đặc trưng thay cho video nặng. Nó cung cấp số lượng lớn ví dụ ngã và không ngã cho classifier. |
| `03_cache_caucafall_kaggle.ipynb` | **Bổ sung dữ liệu có nhãn theo frame và nhiều subject.** Tận dụng PNG/TXT của CAUCAFall để tạo đặc trưng với nhãn chi tiết hơn, đồng thời tránh mở các AVI từng gây lỗi codec. |
| `04_cache_urfd_kaggle.ipynb` | **Bổ sung benchmark fall/ADL dạng chuỗi ảnh.** Chuyển các sequence RGB của URFD thành cùng định dạng feature cache, giúp mô hình học thêm một miền camera và cách dàn dựng khác. |
| `05_cache_mcfd_kaggle.ipynb` | **Bổ sung khả năng chịu thay đổi góc nhìn.** MCFD quay cùng một tình huống từ tám camera; notebook biến các góc nhìn này thành feature nhưng giữ chúng cùng split để phép đánh giá không bị dễ giả tạo. |
| `06_cache_ucf101_kaggle.ipynb` | **Cung cấp hard-negative đa dạng.** UCF101 không phải dữ liệu ngã; vai trò của nó là cho mô hình thấy nhiều hành động bình thường, thể thao và chuyển động mạnh để giảm cảnh báo ngã nhầm. |
| `07_train_temporal_kaggle.ipynb` | **Học và đánh giá hành vi theo thời gian.** Gộp năm feature cache, train classifier đa tỉ lệ + BiGRU + attention do dự án tự xây dựng, chọn threshold và tạo model cùng báo cáo đánh giá cuối. |
| `all_in_one_demo_kaggle.ipynb` | **Kiểm tra nhanh toàn bộ ý tưởng trong một session.** Chạy phiên bản thu nhỏ của các bước trên để xác nhận pipeline hoạt động end-to-end; không thay thế bộ stage khi cần huấn luyện đầy đủ và lưu từng artifact. |

Có thể hiểu ngắn gọn: stage 00 **tổ chức dữ liệu**, stage 01 **học cách nhìn pose**, stage
02-06 **chuyển video thành dữ liệu số nhẹ và đồng nhất**, còn stage 07 **học hành vi ngã theo
thời gian**.

## Luồng của từng notebook

### 00 - `00_build_manifest_kaggle.ipynb`

**Mục đích tổng quát:** chuẩn hóa năm dataset thành một nguồn sự thật duy nhất về file, nhãn
và split. Notebook này không huấn luyện model; nó quyết định dữ liệu nào được phép đi vào từng
giai đoạn sau và ngăn cùng subject/scenario xuất hiện ở cả train lẫn đánh giá.

**Input:** code bundle và đủ năm raw Kaggle Dataset.

**Logic:**

1. Tìm mount ở cả `/kaggle/input/<slug>` và `/kaggle/input/datasets/<owner>/<slug>`.
2. Nhận dạng dataset bằng slug kết hợp cấu trúc thật.
3. Lọc file không canonical, representation processed và bản UCF101 trùng.
4. Kiểm tra số source/group tối thiểu; file dư sinh provenance warning.
5. Chia train/validation/test theo group và stratify nhãn.
6. Kiểm tra mỗi split có cả `fall` và `non-fall`.

**Output:**

- `eldercare_manifest/master_manifest.csv`
- `eldercare_manifest/split_report.json`
- `eldercare_manifest/provenance_audit.json`

### 01 - `01_finetune_pose_kaggle.ipynb`

**Mục đích tổng quát:** tạo một pose extractor dùng chung, phù hợp hơn với góc camera và tư thế
ngã của dữ liệu dự án. Notebook này học vị trí cơ thể, chưa trực tiếp học kết luận fall/non-fall.

**Input:** code bundle, output stage 00, năm raw dataset, `yolov8n-pose.pt`; tùy chọn có
`pose_dataset.yaml` chứa nhãn COCO-17 do người kiểm tra.

**Logic:**

1. Chỉ lấy frame từ classifier training groups để không nhìn validation/test.
2. YOLO-Pose gốc sinh pseudo keypoint labels.
3. Tách pose train/validation theo group, phân tầng và kiểm tra coverage từng dataset family.
4. Lọc pseudo label bằng confidence và số keypoint nhìn thấy tối thiểu.
5. Warm-up pose head 8 epoch trong khi đóng băng 10 layer đầu.
6. Nạp best checkpoint, mở toàn bộ backbone và fine-tune thêm 12 epoch với learning rate thấp.
7. So sánh fitness hai pha và tự rollback về warm-up checkpoint nếu unfreeze làm kết quả xấu đi.
8. `auto` dùng pseudo-only nếu không có gold YAML; nếu có gold thì trộn train nhưng chỉ dùng gold
   validation để chọn model.
9. Tính SHA-256 của checkpoint; chỉ báo cáo gold metric nếu thật sự có gold validation set.

**Output:**

- `yolov8n-pose-finetuned.pt`
- `pose_model.sha256`
- `pose_metadata.json`
- `pose_gold_metrics.json` nếu có gold labels
- pseudo dataset và thư mục run của Ultralytics

Pseudo-only là bootstrap/domain adaptation, không phải bằng chứng pose model tốt hơn trên nhãn
thật.

### 02 - `02_cache_fallvision_kaggle.ipynb`

**Mục đích tổng quát:** biến nguồn video lớn nhất thành chuỗi số nhỏ hơn để những lần train
classifier sau không phải chạy YOLO lại trên hàng nghìn video.

Đọc raw video trong cây `Fall/No_Fall`, bỏ masked/landmarked/processed, chạy pose model và lưu
đặc trưng. Mirror có file vượt inventory FallVision canonical nên phần dư chỉ được xem là
weak-source aggregate.

### 03 - `03_cache_caucafall_kaggle.ipynb`

**Mục đích tổng quát:** đưa nguồn có nhãn frame chi tiết và nhiều người tham gia vào cùng không
gian đặc trưng, giúp classifier học thời điểm/tư thế ngã tốt hơn nhãn video thô.

Đọc trực tiếp PNG, dùng TXT cùng tên để lấy dense frame label `0/1`, và giữ mọi activity của
cùng subject trong một split. AVI không được mở nên tránh lỗi audio/codec từng làm chết kernel.

### 04 - `04_cache_urfd_kaggle.ipynb`

**Mục đích tổng quát:** bổ sung một benchmark fall/ADL độc lập để tăng độ đa dạng miền dữ liệu
và kiểm tra mô hình không chỉ học đặc điểm riêng của FallVision hoặc CAUCAFall.

Đọc trực tiếp các chuỗi `fall-XX-camN-rgb` và `adl-XX-camN-rgb`. Các camera của cùng activity
và sequence dùng chung group.

### 05 - `05_cache_mcfd_kaggle.ipynb`

**Mục đích tổng quát:** cho classifier thấy cùng sự kiện dưới nhiều góc camera, hướng tới khả
năng ít nhạy với vị trí lắp camera mà vẫn tránh đánh giá trùng cùng một tình huống.

Đọc `chute01..24/cam1..8.avi`; tám camera cùng scenario dùng chung group. Scenario 01-22 được
gán weak-positive và 23-24 là confounding/non-fall. Đây là nhãn mức sequence, không phải nhãn
temporal chính xác cho từng frame. Trong hai non-fall scenario, một nằm ở train để model học miền
MCFD và một được giữ nguyên cho test; các camera của cùng scenario không bao giờ bị tách split.

### 06 - `06_cache_ucf101_kaggle.ipynb`

**Mục đích tổng quát:** giảm false alarm bằng cách bổ sung nhiều chuyển động mạnh nhưng không
phải ngã. Notebook này chủ yếu dạy mô hình "những gì không phải fall".

Đọc clip dạng `v_<class>_gXX_cXX`, loại file ngoài schema và clip trùng. Tất cả được gán
non-fall để bổ sung hard-negative; số source/window được giới hạn để không lấn át dữ liệu fall.

### Output chung của stage 02-06

Mỗi cache notebook tạo:

- `pose_cache/*.npz`: feature, frame index, timestamp, frame label và source FPS.
- `shard_manifest.csv`: đường dẫn cache, family, group, split và SHA pose model.
- `shard_metadata.json`: số source, feature dimension, sampling FPS và cấu hình cache.
- `cache_quality_skips.csv`: source bị loại vì quá ít pose frame hoặc detection coverage thấp.

Mỗi frame hợp lệ được chuyển thành vector 89 chiều từ bounding box, 17 keypoint, confidence,
hình học tư thế và thay đổi so với frame trước.

### 07 - `07_train_temporal_kaggle.ipynb`

**Mục đích tổng quát:** sử dụng chuyển động của pose qua nhiều frame để đưa ra quyết định
fall/non-fall cuối cùng, đánh giá quyết định đó và đóng gói model cho ứng dụng.

**Input:** code bundle, output stage 01 và đủ năm output cache của stage 02-06; không cần raw data
hoặc output stage 00. Pose checkpoint stage 01 được kiểm tra SHA rồi đóng gói cùng classifier.

**Logic:**

1. Kiểm tra đủ family, không trùng source, không group leakage và mọi cache có cùng pose SHA,
   feature dimension, sample FPS.
2. Tạo cửa sổ 32 frame, stride 8; nhãn lấy tại tâm cửa sổ.
3. Chia tối đa 10.000 window của mỗi family/split theo nhãn rồi reservoir sampling từng stratum,
   tránh lớp chiếm đa số nuốt toàn bộ ngân sách RAM.
4. Train `MultiScaleTemporalGRU` từ đầu với weighted sampling, temporal/feature masking,
   label smoothing, gradient clipping và cosine learning-rate schedule.
5. Mine 20% negative có xác suất fall cao nhất và fine-tune ngắn với learning rate thấp; chỉ giữ
   checkpoint mới khi validation Average Precision tăng.
6. Chọn epoch bằng Average Precision không phụ thuộc threshold; sau đó tối ưu decision threshold
   bằng ba dự đoán liên tiếp và cooldown giống runtime.
7. Đánh giá test ở mức window, cảnh báo vận hành, source và group/label; đồng thời báo cáo các mức riêng theo dataset.
   Với nguồn chỉ có một lớp, đọc recall hoặc specificity/FPR thay vì F1 hai lớp.
8. Lưu checkpoint và tùy chọn xuất ONNX/OpenVINO.
9. Ghi deployment manifest với pose SHA, sample FPS, sequence length và threshold đã chọn.
   Runtime dùng các giá trị này để giữ cùng độ dài cửa sổ theo cả số frame lẫn thời gian.

**Output:**

- `temporal_attention.pt`
- `training_history.csv`
- `metrics.json`
- `test_predictions.csv`
- `operational_test_predictions.csv`
- `confusion_matrix.csv`
- `deployment_manifest.json`
- `windowing_audit.json`
- `temporal_attention.onnx`
- `temporal_openvino/model.xml`

## Kết quả đạt được sau khi chạy xong stage 00-07

Sau một lượt chạy thành công, dự án có:

1. Một manifest cố định, tái lập được và không leakage theo group.
2. Một YOLO-Pose checkpoint đã domain-adapt bằng pseudo labels, hoặc fine-tune bằng gold +
   pseudo labels nếu đã cung cấp gold dataset.
3. Năm pose-feature cache tái sử dụng được, không cần decode raw video khi train lại classifier.
4. Một classifier temporal hai lớp do dự án tự xây dựng và huấn luyện.
5. Threshold khuyến nghị, metric test, metric theo dataset, confusion matrix và prediction để
   kiểm tra lỗi.
6. Checkpoint PyTorch cùng bản ONNX/OpenVINO phục vụ tích hợp demo Streamlit hoặc benchmark.

Kết quả này đủ cho demo kỹ thuật và baseline dự án cá nhân. Nó chưa chứng minh độ an toàn y tế:
FallVision mirror có provenance phần dư chưa đầy đủ, MCFD dùng weak sequence labels, và phần lớn
fall là tình huống mô phỏng. Cần gold annotation và dữ liệu camera tại môi trường triển khai để
đánh giá trước khi dùng thực tế.

## Notebook demo tùy chọn

**Mục đích tổng quát:** xác nhận nhanh rằng mọi thành phần có thể nối với nhau và tạo ra một
checkpoint hoàn chỉnh trước khi dành nhiều giờ cho full training.

`all_in_one_demo_kaggle.ipynb` chạy cùng logic với giới hạn nhỏ hơn: pose baseline hoặc
fine-tune ngắn, cache ít source, classifier nhỏ và ba epoch. Nó dùng để xác nhận pipeline chạy
end-to-end; output của nó không thay thế kết quả full staged 00-07.
