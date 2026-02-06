# BRAIAIN — Image Sourcing Guide

## How the game images work

- **Real photos** → Hotlinked from Unsplash CDN (`images.unsplash.com/photo-xxx`). Stable, free, attribution-friendly.
- **AI images** → Self-hosted in your repo's `images/` folder. Must be generated or downloaded by you.

## Sourcing AI images (5 per challenge)

Each challenge JSON has `_note` fields on AI rounds telling you exactly what to generate.

### Free generation tools (no watermark, no signup needed)

| Tool | URL | Notes |
|------|-----|-------|
| **Perchance AI** | perchance.org/ai-text-to-image-generator | Unlimited, no watermark, no signup, fast |
| **Upsampler** | upsampler.com/free-image-generator-no-signup | Flux Schnell + Z-Image, no watermark |
| **Vheer** | vheer.com | Free, unlimited, no signup, multiple models |
| **FlatAI** | flatai.org/ai-image-generator-free-no-signup | No watermark, private, local storage |
| **Pixabay AI** | pixabay.com (filter: "AI generated only") | Download existing AI photos — no generation needed |

⚠️ **AVOID**: Raphael.app (adds watermarks), Lexica (requires paid plan for generation)

### Workflow for each challenge

1. Open the challenge JSON (e.g. `challenges/2026-02-09.json`)
2. Find rounds with `"source": "ai"` — each has a `_note` with the prompt
3. Generate or find the image using any tool above
4. Save as the filename specified in `"content"` (e.g. `images/d6r2_ai_pizza_slice.jpg`)
5. Recommended size: 800×600px, JPEG, <200KB

### Pro tips for good AI images

- **Make them tricky**: Use "photorealistic" in your prompts so they're hard to distinguish
- **Vary difficulty**: Make 1-2 obvious AI, 2-3 subtle/hard to detect
- **Match the theme**: Pizza Day = pizza scenes, Valentine's = romantic scenes
- **Include known AI tells**: Hands, text, reflections, backgrounds with people

## Sourcing real photos (5 per challenge)

Use Unsplash — they **require** hotlinking their CDN URLs (not downloading).

### Finding Unsplash photos

1. Search unsplash.com for your theme
2. Click a photo → copy the photo ID from the URL bar
3. Build the CDN URL: `https://images.unsplash.com/photo-{ID}?w=800&h=600&fit=crop&q=80`
4. Add proper attribution: `"Real photograph — [description] by [photographer] on Unsplash"`
5. Link back: `"source_url": "https://unsplash.com/photos/{slug}"`

### URL format reference

```
https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800&h=600&fit=crop&q=80
                                 └─── photo ID ───┘                └── resize params ──┘
```

Parameters: `w` = width, `h` = height, `fit=crop` = crop to fit, `q` = quality (1-100)

## Challenge JSON structure

```json
{
  "day_number": 6,
  "date": "2026-02-09",
  "category": "National Pizza Day",
  "category_emoji": "🍕",
  "rounds": [
    {
      "id": "d6r1",
      "type": "image",
      "content": "https://images.unsplash.com/photo-xxx?w=800&h=600&fit=crop&q=80",
      "source": "human",
      "attribution": "Real photograph — description by Photographer on Unsplash",
      "source_url": "https://unsplash.com/photos/xxx",
      "pro_tip": "Tip shown after answer reveal"
    },
    {
      "id": "d6r2",
      "type": "image",
      "content": "images/d6r2_ai_pizza.jpg",
      "source": "ai",
      "attribution": "AI-generated — description, created with ModelName",
      "pro_tip": "Tip shown after answer reveal",
      "_note": "SELF-HOST: description of what to generate"
    }
  ]
}
```
