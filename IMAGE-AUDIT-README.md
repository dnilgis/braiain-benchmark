# BRAIAIN Image Day — Audit Results & Fix Guide

## What Happened
We audited all 10 Unsplash image URLs for the Images day (Day 2).
- **6 images verified ✅** — real photos by known Unsplash photographers
- **4 images broken ❌** — 3 were AI-generated images that Unsplash purged, 1 was a dead link

## Root Cause
Unsplash has been actively removing AI-generated content from their platform.
Three of the broken images had October 2023 timestamps — a period when many
AI-generated images were uploaded before Unsplash tightened enforcement.

**Lesson: Never use Unsplash to host AI-generated images for the game.**
Unsplash is great for "human" rounds. For "AI" rounds, self-host the images.

## Verified Working Images (Unsplash)

| Round | Photo ID | Photographer | Subject |
|-------|----------|-------------|---------|
| 1 | `photo-1481349518771-20055b2a7b24` | Susan Yin | Library bookshelves |
| 2 | `photo-1485827404703-89b55fcc595e` | Alex Knight | Humanoid robot |
| 4 | `photo-1512621776951-a57141f2eefd` | Anna Pelzer | Fresh vegetables |
| 5 | `photo-1526374965328-7f61d4dc18c5` | Markus Spiske | Matrix code rain |
| 6 | `photo-1558618666-fcd25c85f82e` | Pawel Czerwinski | Colorful abstract (**NEW**) |
| 8 | `photo-1531297484001-80022131f5a1` | Clint Patterson | Laptop purple glow |
| 10 | `photo-1550745165-9bc0b252726f` | Lorenzo Herrera | Retro gaming neon |

## Self-Hosted AI Images Needed

Create an `images/` folder in your repo and add these:

| Round | Filename | What to Generate |
|-------|----------|-----------------|
| 3 | `d2r3_ai_abstract_landscape.jpg` | Dreamy AI landscape — rolling hills, impossible colors, slightly surreal sky |
| 7 | `d2r7_ai_dreamlike_cityscape.jpg` | AI cityscape — futuristic buildings, garbled text on signs, floating elements |
| 9 | `d2r9_ai_surreal_architecture.jpg` | AI architecture — impossible building with physics-defying cantilevers |

### Where to generate them
- **Midjourney** (best quality, most realistic)
- **DALL-E 3** via ChatGPT (free with Plus)
- **Stable Diffusion** via [Lexica.art](https://lexica.art) (free, can download existing ones)

### Tips for good AI images for the game
- Pick images that are ALMOST convincing but have subtle tells
- Avoid obviously broken hands/faces (too easy to spot)
- Look for: garbled text, impossible reflections, repetitive patterns, physics violations
- Save at 800px wide, JPEG quality 80+

## Updated daily.json
See `daily-images.json` — drop-in replacement with all fixes applied.
Update the `content` paths for rounds 3, 7, 9 once you've generated the AI images.
