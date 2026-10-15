# Bokin FX

App thêm hiệu ứng vào video loop rồi xuất video **30 giây hoặc 1 phút**, vẫn loop liền mạch. Giao diện giống Adobe
Premiere Pro:

- **Thư viện hiệu ứng**: nhập video hiệu ứng có sẵn (nền đen hoặc nền trong suốt) và ảnh PNG. Thư viện dùng chung cho
  mọi dự án.
- **Timeline**: kéo hiệu ứng xuống track, đặt trong một đoạn thời gian, kéo, cắt bớt, cắt đôi, mờ dần vào/ra.
- **Xem nhanh**: bấm phát là thấy ngay kết quả. Bật nút **🎯** để khi dừng hình, app hiện khung chính xác như video sẽ
  xuất ra.
- **Vùng đứng yên**: tô chỗ nào trong hình phải đứng yên tuyệt đối. Ở đó app lấy hình từ ảnh gốc, hoặc từ một khung
  của video.
- **Vùng cho từng clip**: mỗi clip có vùng riêng (tô bằng cọ hoặc khoanh), mép mềm chỉnh được. Phóng to hình để
  khoanh cho kỹ.
- **Tiền cảnh**: tô những vật ở gần (khung cửa, cây, mái nhà). Clip bật "Bay ra sau tiền cảnh" sẽ bị chúng che.
- **Hiệu ứng tạo sẵn**: tuyết, lá rơi, mưa, đom đóm / bụi sáng. App tự vẽ, chỉnh được mật độ, cỡ, gió…

## Cài đặt

1. Vào trang [Releases](https://github.com/vhung-spec/bokin-video-fx/releases), tải file `BokinFX-Setup-<phiên bản>.exe` mới nhất.
2. Bấm đúp vào file để cài. Không cần quyền quản trị.
3. Mở **Bokin FX** từ menu Start hoặc biểu tượng ngoài màn hình. App mở trong một cửa sổ riêng (Edge hoặc Chrome).

App tự kiểm tra bản mới khi mở cửa sổ và mỗi 5 phút. Muốn kiểm tra ngay thì bấm vào dòng **phiên bản** ở góc
dưới. Khi có bản mới, góc trên có nút **Cập nhật ngay**. Nếu không bấm, bản mới sẽ được cài lúc app tự đóng.

## Màn hình

| Vùng | Dùng để |
|---|---|
| Trên bên trái: **Bảng chỉnh hiệu ứng** | Chỉnh clip đang chọn: thời gian, mờ dần, hoà trộn, vị trí, cỡ, vùng |
| Trên bên phải: **Chương trình** | Xem video. Nút phát, khung trước/sau, phóng to hình. Cột **Công cụ** bên phải: vẽ vùng, vị trí và cỡ |
| Dưới bên trái: **Thư viện / Tạo sẵn / Dự án** | Các hiệu ứng để kéo xuống timeline. Các dự án đã mở |
| Dưới bên phải: **Timeline** | V1 là video gốc (lặp lại thành 30 giây hoặc 1 phút). V2, V3… là các track hiệu ứng |

Kéo đường kẻ giữa các vùng để đổi cỡ.

## Cách dùng

1. **Mở video**: bấm **+ Video mới…** hoặc kéo file video vào màn hình. Thường là video loop 6 giây (1920×1080).
2. **Chọn độ dài**: trên thanh Timeline, bấm **30 giây** hoặc **1 phút**.
3. **Nhập hiệu ứng**: ở thẻ **Thư viện**, bấm **+ Nhập file…** (hoặc kéo file vào). Chọn video hiệu ứng hoặc ảnh PNG.
4. **Thêm vào video**: kéo hiệu ứng từ Thư viện xuống một track trên timeline. Clip chạy từ chỗ thả đến hết video.
   Bấm đúp vào hiệu ứng để thêm suốt video.
5. **Chỉnh clip**: bấm vào clip trên timeline, rồi chỉnh ở **Bảng chỉnh hiệu ứng**:
   - **Thời gian**: bắt đầu, kết thúc, mờ dần vào/ra. Gõ số giây (`2.5` hay `2,5` đều được), rồi bấm Enter.
   - **Hoà trộn**: video hiệu ứng trên nền đen dùng **Screen**. File nền trong suốt dùng **Bình thường**. App tự chọn
     khi bạn thêm file.
   - **Vị trí và cỡ**, **Độ đậm**. Với video/ảnh ngoài, chọn clip là trên màn hình hiện một khung trắng: kéo bên trong
     khung để di chuyển, kéo một góc để phóng to/thu nhỏ, bấm đúp để đưa về giữa. Giữ `Shift` để đi thẳng. Bấm vào
     hình của một clip trên màn hình để chọn clip đó; bấm rồi kéo luôn để di chuyển nó. Phóng to màn hình để đặt
     clip thật chính xác (mỗi bước nhích khoảng một điểm ảnh).
   - **Vùng**: bấm **✎ Vẽ vùng** để tô chỗ hiệu ứng được hiện.
6. **Vùng đứng yên và tiền cảnh**: bấm vào **V1 Video gốc** trên timeline, rồi bấm **✎ Vẽ vùng đứng yên** hoặc
   **✎ Vẽ tiền cảnh**.
   - **Đổi video gốc**: bấm vào **V1 Video gốc** rồi **Đổi video gốc…**, hoặc kéo file video thả vào hàng V1.
     Timeline, các clip và vùng đã vẽ được giữ nguyên (thời gian clip giữ theo giây). Clip nằm sau cuối video mới
     được dời lên trước. Ảnh gốc của vùng đứng yên được khớp lại theo cỡ video mới.
7. **Xem**: bấm **▶** (hoặc phím cách). Dừng thì hình đứng yên đúng khung đang thấy. Muốn xem đúng như video sẽ
   xuất ra, bật **🎯 Khung chính xác** rồi dừng hình (app dựng khung đó, mất chút thời gian).
8. **Xuất video**: bấm **Xuất video** ở góc trên bên phải. Video nằm trong thư mục **Videos\Bokin FX**.

### Timeline

| Thao tác | Cách làm |
|---|---|
| Di chuyển clip | Kéo clip. Kéo lên hàng **+ track mới** để tạo track mới |
| Cắt bớt đầu / cuối | Kéo mép trái / phải của clip |
| Cắt đôi | Chọn dao cắt ✂ (phím `C`) rồi bấm vào clip. Hoặc `Ctrl+K`: cắt clip đang chọn tại đầu phát (không chọn clip nào thì cắt mọi clip ở đầu phát). Không cắt ở chỗ clip đang mờ dần |
| Nhân đôi | Giữ `Alt` rồi kéo clip, hoặc bấm chuột phải vào clip |
| Xoá clip | Chọn clip rồi bấm `Delete` |
| Hoàn tác / làm lại | `Ctrl+Z` / `Ctrl+Shift+Z` |
| Ẩn một track | Bấm 👁 ở đầu track |
| Phóng to timeline | Kéo thanh trượt ở góc phải, phím `+` `-`, hoặc `Ctrl` + lăn chuột. Phím `\` để xem cả timeline |
| Bám dính | Nút 🧲 (phím `S`): clip bám vào đầu phát, mép clip khác và chỗ nối loop |

Clip chạy suốt video tự nối liền mạch ở chỗ video lặp lại. Clip ngắn hơn luôn mờ dần ở đầu và cuối video, để chỗ
nối không bị giật.

### Vẽ vùng

Bấm một nút trong cột **Công cụ** bên phải màn hình (Vùng đứng yên, Tiền cảnh, Vùng của clip đang chọn). Các công cụ vẽ
hiện trong cột đó, hình không bị thu nhỏ.

| Thao tác | Cách làm |
|---|---|
| Tô | Chọn **🖌 Cọ** rồi kéo chuột. Chỉnh **Cỡ**, hoặc bấm phím `[` `]` |
| Khoanh vùng | Chọn **⬠ Khoanh**, bấm từng điểm, rồi bấm lại điểm đầu (hoặc bấm đúp, hoặc Enter) |
| Thêm / bớt | **＋ Thêm**, **－ Bớt** (phím `X` để đổi) |
| Hoàn tác / làm lại | `Ctrl+Z` / `Ctrl+Y` |
| Phóng to, kéo hình | Lăn chuột trên hình để phóng to. Giữ chuột phải (hoặc phím cách) rồi kéo |
| Xoá cả vùng | **Xoá vùng** (vùng đứng yên, tiền cảnh) hoặc **Cả khung hình** (vùng của clip). Bấm `Ctrl+Z` để lấy lại |
| Xong | Bấm **Xong** hoặc `Esc`. Vùng của clip vẫn hiện bằng viền xanh để bạn kéo clip vào đúng chỗ (tắt bằng ô **Hiện vùng của clip**) |

Vùng của một clip lúc đầu là cả khung hình. Nét tô (hoặc vùng khoanh) đầu tiên thay cả khung hình bằng chỗ bạn tô.

### Phím tắt khác

`Space` phát/dừng · `←` `→` khung trước/sau (`Shift`: 5 khung) · `↑` `↓` về mép clip trước/sau · `Home` `End`
về đầu/cuối · `Ctrl+M` xuất video.

## Bản quyền hình và video

Chỉ dùng video, ảnh và hiệu ứng mà công ty sở hữu hoặc có giấy phép. Hình lá, tuyết, mưa, đom đóm của app đều do mã
tự vẽ, không lấy từ đâu.

## Dữ liệu nằm ở đâu

- Dự án (video đã mở, vùng đã vẽ, timeline): `%APPDATA%\Bokin\FX\projects`
- Thư viện hiệu ứng: `%APPDATA%\Bokin\FX\library`
- Nhật ký khi có lỗi: `%APPDATA%\Bokin\FX\app.log`
- Video đã xuất: `Videos\Bokin FX`

Gỡ cài đặt hoặc cập nhật không xoá những thư mục này. Dự án làm bằng bản cũ được tự chuyển sang timeline khi mở.

## Cho người phát triển

- Chạy từ mã nguồn: `pip install -r requirements.txt`, rồi `python bokin_fx.py`.
- Kiểm thử: `python -m unittest discover -s tests -t .`
- Tự kiểm tra trọn vẹn (làm một video thật): `python bokin_fx.py --selftest`
- Thử giao diện trong trình duyệt (cần `pip install playwright`): `python packaging/ci_page_test.py`
- Cấu trúc và quy tắc: [CLAUDE.md](CLAUDE.md).
- Mỗi lần push lên nhánh phát hành, GitHub Actions build trên Windows, chạy kiểm thử, thử giao diện trên Edge, cài
  thử, thử tự cập nhật rồi phát hành bản cài mới (`.github/workflows/bokin-fx.yml`).
