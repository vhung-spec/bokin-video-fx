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
    (373.8, "7. Vùng sáng nhấp nháy"), (441.68, "8. Sao chép thiết lập và vùng"),
    (529.04, "9. Thư viện: file hiệu ứng"), (581.92, "10. Sửa trên timeline"), (645.6, "11. Xem cho kỹ"),
    (675.28, "12. Lưu dự án"), (690.52, "13. Xuất video"), (724.48, "14. Phím tắt và Hướng dẫn"),
    (746.8, "15. Chó ngủ thở"), (874.44, "Kết thúc"))
# (part 9, Lửa lò sưởi, taken out 2026-11-02 with the effect: the parts after it renumbered, on their cards too)
LENGTH = 891.28  # (seconds)


def video_path():
    """HUONG-DAN.mp4: in the bundle next to index.html, or at the top of the source folder."""
    here = os.path.join(paths.resource_dir(), VIDEO)
    return here if os.path.exists(here) else os.path.join(os.path.dirname(paths.resource_dir()), VIDEO)


def view():
    """For the page: whether the video is there, and its chapters."""
    return {"ready": os.path.exists(video_path()), "length": LENGTH,
            "chapters": [{"at": at, "title": title} for at, title in CHAPTERS]}
