"""The video guide ("Video hướng dẫn"; the owner, 2026-10-25: "chèn video hướng dẫn sử dụng vào trong app"):
HUONG-DAN.mp4 beside HUONG-DAN.md (in the bundle next to index.html, as keys.guide_path finds the written guide),
played in the page with its chapters beside it. Silent, every step written on the screen: the owner chose it without
the voice (the voice we had was made by a model whose training data is licensed for research only). 1600x900 H.264
(the app's window is Edge or Chrome), its index at the front of the file (it plays at once, any chapter), no tool's
name in it."""
import os

from . import paths

VIDEO = "HUONG-DAN.mp4"
CHAPTERS = (  # (where each part starts, seconds; its title in the page's list)
    (0.0, "Mở đầu"), (12.0, "1. Kích hoạt app"), (42.8, "2. Giao diện"), (73.48, "3. Mở video loop"),
    (100.52, "4. Vùng đứng yên"), (170.76, "5. Tuyết rơi ngoài cửa sổ (vẽ vùng cho hiệu ứng)"), (282.96, "6. Dây đèn"),
    (373.8, "7. Vùng sáng nhấp nháy"), (441.68, "8. Sao chép thiết lập và vùng"), (529.04, "9. Lửa lò sưởi"),
    (594.2, "10. Thư viện: file hiệu ứng"), (647.08, "11. Sửa trên timeline"), (710.76, "12. Xem cho kỹ"),
    (740.44, "13. Lưu dự án"), (755.68, "14. Xuất video"), (789.64, "15. Phím tắt và Hướng dẫn"), (811.96, "Kết thúc"))
LENGTH = 826.56  # (seconds)


def video_path():
    """HUONG-DAN.mp4: in the bundle next to index.html, or at the top of the source folder."""
    here = os.path.join(paths.resource_dir(), VIDEO)
    return here if os.path.exists(here) else os.path.join(os.path.dirname(paths.resource_dir()), VIDEO)


def view():
    """For the page: whether the video is there, and its chapters."""
    return {"ready": os.path.exists(video_path()), "length": LENGTH,
            "chapters": [{"at": at, "title": title} for at, title in CHAPTERS]}
