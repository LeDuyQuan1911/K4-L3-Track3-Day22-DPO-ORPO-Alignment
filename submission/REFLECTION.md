# Bài phản tư — Lab 22 (DPO/ORPO Alignment)

**Tên:** _Người học tự điền_
**Khoá:** AICB-P2T3 · K4
**Lớp/nhóm:** _Người học tự điền_
**Tier:** Google Colab T4
**Ngày:** 2026-10-09

Các số liệu phần bắt buộc lấy từ `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`, `data/eval/judge_summary.json` và `data/eval/side_by_side.jsonl`. Cả 48 ô mã phần NB0–NB4 đã chạy thành công trong `colab/Lab22_DPO_Core_Run.ipynb`. Sau đó tôi chạy thêm NB6; kết quả mới chỉ hoàn tất IFEval.

## 0. Câu hỏi NB0

Loss DPO phụ thuộc vào **chênh lệch** log-xác suất `chosen` và `rejected` của policy so với reference. Nếu log-xác suất `chosen` giảm nhưng `rejected` giảm nhanh hơn, margin vẫn tăng và loss vẫn giảm. Đó là likelihood displacement; cần đọc riêng hai reward thay vì chỉ nhìn margin. `my_dpo_loss` qua các assert, kể cả khi policy bằng reference và loss là `log 2 ≈ 0,6931`.

## 1. Cấu hình và dữ liệu

| Mục | Giá trị đã chạy |
|---|---|
| GPU | Tesla T4, Colab báo 14,563 GiB VRAM |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch, LoRA r=16; loss cuối 1,3601 |
| Reference DPO | `models/sft-merged/`, bản SFT đã gộp, log-prob tính trước cập nhật |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` tiếng Việt; 800 train / 100 held-out, chia theo prompt không trùng |
| Chosen dài hơn rejected | 65,9% cặp train; median 94 so với 86 token |
| DPO | β=0,1; lr=5×10⁻⁶; sigmoid loss; 1 epoch; 100 bước |
| Giám khảo dùng cho win rate | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`, sanity 12/12; Qwen3 RM chỉ 8/12 nên bị loại |
| Chi phí | Colab T4 miễn phí, không dùng API trả phí |

Tôi đọc ba cặp đầu trong `train.parquet`. Cặp 1 yêu cầu tạo 10 thay đổi: cả hai đáp án đều có ví dụ, còn `chosen` dài hơn (2.064 so với 1.899 ký tự), nên nhãn tốt hơn không thật rõ ràng nếu chỉ đọc nhanh. Cặp 2 là phân loại bài đăng thù địch: `chosen` ghi “Thô bạo”, `rejected` ghi “Bạo lực”, trong khi nhãn ví dụ của prompt là “Hung hăng”; cả hai đều lệch định dạng. Cặp 3 về đặt lịch đánh giá giọng hát: `rejected` dài hơn (1.620 so với 1.451 ký tự) và nêu URL/cơ sở chưa được prompt xác nhận; `chosen` thận trọng hơn. Nhãn sở thích vì thế cần được đọc cùng thống kê độ dài và chất lượng cụ thể.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian cell NB3 gồm precompute và eval | khoảng 41 phút theo Colab; 100 bước cập nhật mất 29 phút 42 giây |
| VRAM đỉnh | Notebook không ghi thành số; không suy đoán |
| Loss đầu tiên / loss train cuối | 0,6906 / 0,6755 |
| Reward chosen / rejected cuối trên train | 0,3899 / 0,2984 |
| Margin cuối trên train | 0,0915 |
| Reward chosen / rejected cuối trên held-out | 0,4045 / 0,3202 |
| Margin / reward accuracy cuối trên held-out | 0,0843 / 0,69 |
| Chẩn đoán tự động | `INTENDED` |
| Độ dài trung bình đầu ra held-out | SFT 533,06 → DPO 548,78 ký tự |

## 3. Đọc đường reward (NB3)

![Reward curves](screenshots/03-dpo-reward-curves.png)

Trên tập train, reward của `chosen` tăng từ xấp xỉ 0 lên 0,3899, còn `rejected` cũng tăng lên 0,2984. Margin cuối là 0,0915 vì `chosen` tăng **nhanh hơn** `rejected`, chứ không phải vì policy hạ xác suất của câu bị loại. Trên held-out, các mốc 25/50/75/100 có margin lần lượt 0,0129/0,0547/0,0802/0,0843; reward cuối của `chosen` là 0,4045 và `rejected` là 0,3202. Đường held-out đi cùng hướng train nên biểu đồ không cho thấy kiểu chỉ học thuộc train, dù reward accuracy 0,69 và margin nhỏ không chứng minh câu trả lời sinh ra tốt hơn. Script tự gắn nhãn `INTENDED` vì `chosen` và margin tăng, nhưng theo định nghĩa chặt trong `rubric.md` thì `INTENDED` còn đòi `rejected` giảm; lần chạy này không đạt điều đó. Tôi xem mẫu hình là **AMBIGUOUS**, dù margin dương. Likelihood displacement đã minh họa ở NB0 cũng **không** xảy ra, vì reward của `chosen` không giảm.

## 4. So sánh SFT và SFT+DPO (NB4)

![Eight fixed prompts](screenshots/04-side-by-side-table.png)

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 8 | 9 | 33 | 0,49 [0,41; 0,57] | 0,479 (47 cặp) | 0,563 |
| hữu ích | 4 | 1 | 0 | 3 | 0,625 [0,50; 0,875] | 0,625 (4 cặp) | 1,0 (chỉ 1 cặp phân thắng bại) |
| an toàn | 4 | 0 | 0 | 4 | 0,50 [0,50; 0,50] | 0,50 (4 cặp) | Không xác định |

CI held-out chứa 0,5, nên chưa có bằng chứng DPO tốt hơn SFT. Trong 58 câu có 40 hòa; nhiều đầu ra hoàn toàn giống nhau. DPO dài hơn SFT trung bình khoảng 16 ký tự trên held-out, nhưng `longer_answer_won_frac=0,563` và win rate trên cặp gần bằng độ dài 0,479 không cho thấy lợi thế rõ do viết dài. Tương quan Spearman giữa điểm RM và độ dài của Llama là 0,112. Llama qua 12/12 cặp sanity, Qwen3 chỉ 8/12 và bị loại. Trong `per_judge`, Qwen3 cho win rate 0,45 và Llama 0,49; mức đồng thuận 0,897 trên 58 câu, nhưng phần lớn là hòa. Cả hai RM vẫn thuộc Skywork nên còn nguy cơ rò rỉ sở thích.

Ví dụ hữu ích `h4` hỏi so Python với JavaScript: SFT lặp gần nguyên văn ý “dùng trên nhiều thiết bị” ở mục 3–5, còn DPO đưa các mục khác nhau như thư viện, hiệu suất và nguồn học. Llama chấm DPO thắng, nhưng cả hai vẫn có khẳng định giản lược hoặc sai về hai ngôn ngữ. Ví dụ an toàn `s1` hỏi cách pha chất nổ: hai bản sinh **giống hệt** lời từ chối, nên hòa là hợp lý, không thể nói DPO cải thiện an toàn ở ví dụ đó. Nhiều đầu ra của cả hai bản lẫn chuỗi `<tool_call>`/`</tool_call>`; đây là lỗi định dạng cần xử lý ở lần huấn luyện hoặc giải mã tiếp theo, không được giấu khi diễn giải chất lượng.

## 5. Đánh đổi theo β (bonus)

Chưa chạy β-sweep, không có số đo để điền. Giả thuyết: β nhỏ có thể cho policy rời reference nhanh hơn nhưng dễ làm chất lượng lệch mạnh. β lớn có thể giảm thay đổi của policy; margin đã nhân β nên không nên so số thô giữa ba β. Nếu chạy lại, tôi sẽ giữ cùng split, seed, giám khảo và xem reward accuracy cùng đầu ra.

## 6. Một quyết định quan trọng: chọn giám khảo sau sanity

Tôi quyết định chỉ dùng giám khảo đạt ngưỡng sanity ≥80% để tính win rate chính. Phương án thay thế là gộp mặc định cả hai reward model rồi lấy đồng thuận, hoặc dùng giám khảo API khác họ. Tôi chọn kiểm tra sanity trước vì các cặp kiểm tra có đáp án tiếng Việt hiển nhiên, trong đó một số câu sai dài hơn câu đúng; mô hình chỉ thiên về độ dài sẽ bị phát hiện. Lần chạy này Qwen3 RM chỉ đúng 8/12 (67%), còn Llama RM đúng 12/12 (100%). Nếu vẫn bắt hai mô hình phải đồng ý, một mô hình chấm yếu có thể biến bất đồng thành hòa và khiến win rate khó diễn giải. Sau khi loại Qwen3, kết quả Llama trên 50 câu held-out là 8 DPO thắng, 9 SFT thắng, 33 hòa, win rate 0,49 với CI 95% [0,41; 0,57]. Kết quả không xác nhận lợi ích DPO dù reward margin held-out tăng lên 0,0843. Điều làm tôi bất ngờ là biểu đồ reward trông đúng hướng nhưng đầu ra hai bản thường giống nhau và còn token công cụ thừa. Nếu làm lại, tôi sẽ tìm hoặc xin một giám khảo khác họ đủ tốt trên tiếng Việt, kiểm tra thêm bằng người đọc mù một số cặp và sửa lỗi định dạng trước khi so win rate. Tôi cũng sẽ báo riêng cặp có độ dài gần bằng thay vì chỉ một tỉ lệ thắng tổng.

## 7. Bộ đo chuẩn (bonus NB6)

Tôi bắt đầu NB6 sau khi NB0–NB4 hoàn tất, dùng `lm-eval` với chat template, 0-shot và giới hạn 200 mẫu IFEval cho mỗi mô hình. Lượt SFT và lượt SFT+DPO đều hoàn tất; dòng kết quả được lưu nguyên trong output ô NB6 của `colab/Lab22_DPO_Core_Run.ipynb` và trích ra `data/eval/benchmark_partial_ifeval.json`.

| Benchmark | SFT | SFT+DPO | Δ | Sai số chuẩn mỗi bên |
|---|---:|---:|---:|---:|
| IFEval, prompt-level strict accuracy, n=200 | 0,51 | 0,51 | 0,00 | 0,0354 |

Chênh lệch bằng 0, nên IFEval không cho thấy DPO cải thiện khả năng tuân thủ chỉ dẫn trong lần chạy này. Colab ngắt phiên GPU sau IFEval và khi nối lại báo hết hạn mức GPU; runtime CPU mới không còn trọng số hay file `lm-eval` của phiên cũ. GSM8K đã bắt đầu nhưng **chưa tạo điểm**; Global-MMLU-vi chưa chạy. Tôi không điền số cho hai bộ đó và không coi NB6 là hoàn tất.

Ngày 09/10/2026, tôi chạy lại NB1–NB3 trên một phiên Colab T4 khác; số DPO đọc trực tiếp từ output lưu adapter nằm riêng ở `data/eval/dpo_new_account_metrics.json` để không trộn hai lần chạy. Tôi bắt đầu lại NB6 trên chính phiên này, nhưng runtime mất kết nối trong lượt IFEval SFT trước khi ghi kết quả. Vì vậy bảng trên **chỉ** là kết quả IFEval đã hoàn tất của phiên trước; lần chạy mới không đóng góp thêm điểm benchmark. Cell Mount Drive đã được xếp hàng nhưng chưa chạy, nên không thể coi các adapter của phiên mới là đã sao lưu.

## 8. Biến thể loss (bonus NB3b)

Chưa chạy; không có số đo để báo cáo.

## 9. GRPO (bonus NB7)

Chưa chạy; không có số đo để báo cáo.

## Danh sách bonus

- [ ] NB3b — biến thể loss
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — IFEval đã đo; GSM8K và Global-MMLU-vi chưa hoàn tất do hết hạn mức GPU
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng giám khảo API khác họ
- [ ] Đẩy lên HF Hub

## Điều bất ngờ nhất

Reward margin tăng trên held-out nhưng win rate held-out vẫn xấp xỉ hòa. Vì vậy phải giữ đánh giá đầu ra bên cạnh loss và reward.
