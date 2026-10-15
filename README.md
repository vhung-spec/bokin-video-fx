# Bokin FX

App thêm hiệu ứng vào video loop, giống vài chức năng của After Effects:

- **Hiệu ứng hạt**: tuyết rơi, lá rơi (lá thu, lá xanh, cánh hoa anh đào), mưa, hạt sáng (đom đóm, bụi sáng lơ lửng).
- **Video / ảnh ngoài**: ghép video hiệu ứng có sẵn (nền đen hoặc nền trong suốt) hoặc ảnh PNG, với chế độ hoà trộn
  Normal, Screen, Add, Multiply, chỉnh độ đậm, cỡ, vị trí.
- **Vùng đứng yên**: tô chỗ nào trong hình phải đứng yên tuyệt đối. Ở đó app lấy hình từ ảnh gốc, hoặc từ một khung của video.
- **Vùng cho từng hiệu ứng**: mỗi lớp hiệu ứng có vùng riêng (tô bằng cọ hoặc khoanh vùng), mép mềm chỉnh được.
- **Tiền cảnh**: tô những vật ở gần (khung cửa, cây, mái nhà). Hiệu ứng bật "Bay ra sau tiền cảnh" sẽ bị chúng che.

Video ra **vẫn loop liền mạch**: mọi chuyển động thêm vào lặp lại đúng sau một vòng loop. Video ngoài dài hay ngắn
hơn loop đều được: app tự hoà mượt chỗ nối.

## Cài đặt

1. Vào trang [Releases](https://github.com/vhung-spec/bokin-video-fx/releases), tải file `BokinFX-Setup-<phiên bản>.exe` mới nhất.
2. Bấm đúp vào file để cài. Không cần quyền quản trị.
3. Mở **Bokin FX** từ menu Start hoặc biểu tượng ngoài màn hình. App mở trong một cửa sổ riêng (Edge hoặc Chrome).

App tự kiểm tra bản mới khi mở và mỗi 6 giờ. Khi có bản mới, góc trên có nút **Cập nhật ngay**. Nếu không bấm,
bản mới sẽ được cài lúc app tự đóng.

## Cách dùng

1. **Mở video**: bấm **+ Mở video…** hoặc kéo file video vào cửa sổ. Thường là video loop 6 giây (1920×1080, 24 khung/giây).
2. **Vùng đứng yên** (bảng bên phải, mục "Ảnh nền"):
   - Chọn nguồn: một khung của video, hoặc bấm **Chọn ảnh gốc…** để dùng ảnh gốc.
   - Vào thẻ **Vẽ vùng** và tô chỗ phải đứng yên.
3. **Thêm hiệu ứng**: bấm **+ Tuyết rơi**, **+ Lá rơi**, **+ Mưa**, **+ Hạt sáng** hoặc **+ Video / ảnh ngoài**.
   - Kéo các thanh để chỉnh mật độ, cỡ, tốc độ, gió, độ đậm…
   - **Đổi kiểu ngẫu nhiên**: giữ thông số, đổi vị trí các hạt.
   - Lớp trên cùng trong danh sách sẽ phủ lên trên. Dùng ▲ ▼ để đổi thứ tự.
4. **Chỉ cho hiệu ứng ở một chỗ**: chọn lớp, vào **Vẽ vùng**, bấm **Xoá hết** rồi tô chỗ muốn có hiệu ứng.
   Hoặc chọn **－ Bớt** để xoá những chỗ không cần.
5. **Video / ảnh ngoài**: thêm lớp, bấm **Chọn file…**.
   - Video hiệu ứng trên nền đen: dùng chế độ **Screen**.
   - File có nền trong suốt (MOV ProRes 4444, WebM, PNG): dùng **Bình thường**. App tự chọn khi bạn thêm file.
   - **Hoà mượt chỗ nối loop** (giây): độ dài đoạn chuyển mượt ở chỗ nối.
6. **Xem kết quả**: thẻ **Xem kết quả** hiện một khung với mọi hiệu ứng. Kéo thanh thời gian để xem khung khác.
7. **Video xem thử**: bấm **▶ Làm video xem thử**. App làm một bản nhỏ của cả vòng loop và phát lặp lại để bạn xem chỗ nối.
8. **Xuất video**: chọn độ dài (1 vòng, 30 giây, 1 phút… tới 12 giờ) rồi bấm **Xuất video**.
   - Video nằm trong thư mục **Videos\Bokin FX**. Bấm **Mở thư mục video** để mở.
   - Video dài được nối từ cùng một vòng loop mà không nén lại, nên không mất chất lượng và không có chỗ nối mới.

### Vẽ vùng

| Thao tác | Cách làm |
|---|---|
| Tô | Chọn **🖌 Cọ** rồi kéo chuột. Chỉnh **Cỡ cọ**, hoặc bấm phím `[` `]` |
| Khoanh vùng | Chọn **⬠ Khoanh vùng**, bấm từng điểm, rồi bấm lại điểm đầu (hoặc bấm đúp, hoặc Enter) |
| Thêm / bớt | **＋ Thêm**, **－ Bớt** (phím `X` để đổi) |
| Hoàn tác / làm lại | `Ctrl+Z` / `Ctrl+Y` |
| Phóng to, kéo hình | Lăn chuột để phóng to. Giữ chuột phải (hoặc phím cách) rồi kéo |
| Khung trước / sau | Phím `←` `→` |

## Bản quyền hình và video

Chỉ dùng video, ảnh và hiệu ứng mà công ty sở hữu hoặc có giấy phép. Hình lá, tuyết, mưa, đom đóm của app đều do mã
tự vẽ, không lấy từ đâu.

## Dữ liệu nằm ở đâu

- Dự án (video đã mở, vùng đã vẽ, thiết lập): `%APPDATA%\Bokin\FX\projects`
- Nhật ký khi có lỗi: `%APPDATA%\Bokin\FX\app.log`
- Video đã xuất: `Videos\Bokin FX`

Gỡ cài đặt hoặc cập nhật không xoá những thư mục này.

## Cho người phát triển

- Chạy từ mã nguồn: `pip install -r requirements.txt`, rồi `python bokin_fx.py`.
- Kiểm thử: `python -m unittest discover -s tests -t .`
- Tự kiểm tra trọn vẹn (làm một video thật): `python bokin_fx.py --selftest`
- Cấu trúc và quy tắc: [CLAUDE.md](CLAUDE.md).
- Mỗi lần push lên nhánh phát hành, GitHub Actions build trên Windows, chạy kiểm thử, cài thử, thử tự cập nhật rồi phát
  hành bản cài mới (`.github/workflows/bokin-fx.yml`).
