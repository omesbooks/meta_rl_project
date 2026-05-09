# 🎨 Assets — Logo & Icon Files

Put your logo / icon files here. Then reference them in `rl_app.py` BRANDING:

```python
BRANDING = {
    "window_icon":     "assets/icon.ico",     # Window titlebar/taskbar
    "logo_image":      "assets/logo.png",     # Sidebar logo
    "logo_image_size": (32, 32),              # Pixel size (w, h)
    ...
}
```

## Recommended formats

| File | Format | Size | Notes |
|---|---|---|---|
| Window icon | `.ico` (Windows) or `.png` | 32x32, 64x64, 128x128 | Multiple sizes in .ico best |
| Sidebar logo | `.png` (transparent) | Any (auto-resized to logo_image_size) | PNG with alpha channel |

## Example files

Place your files here:
```
assets/
├── logo.png       (sidebar logo)
├── icon.ico       (window icon, Windows preferred)
├── icon.png       (window icon, fallback)
└── README.md      (this file)
```

## Free icon resources
- https://flaticon.com (search "trading", "AI", "robot")
- https://icons8.com
- https://emojipedia.org (download as PNG)

## Image requirements
- PNG with transparency works best for sidebar
- Square aspect ratio recommended (e.g., 64x64, 128x128)
- File size < 100 KB ideal
